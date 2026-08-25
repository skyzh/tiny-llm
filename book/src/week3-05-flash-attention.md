# 🚧 Week 3 Day 5: Paged FlashAttention

> 🚧 This chapter is under review and may change.

In this chapter, we will tile page-aware attention for multi-token queries.
The operator translates logical K/V positions through `block_table`, stages
page-backed tiles on chip, and combines them with online softmax. Short queries
continue to use the vector decode schedule from Day 4; long prefill chunks use
the tiled schedule developed here.

This is a required chapter. FlashAttention belongs here rather than in Week 2
because the serving model's real K/V source is now the page pool. Building a
dense-only kernel first would create a second attention path and then require
students to relearn its memory schedule around page translation.

## Prerequisites

This chapter combines four prerequisites:

- Week 2 Day 5 introduced the online-softmax recurrence.
- Week 2 Day 6 introduced the cooperative 32×32 tile built from BF16 8×8
  SIMD-matrix fragments.
- Week 3 Day 3 introduced physical pages and block tables.
- Week 3 Day 4 introduced direct page-walking attention and the decode
  schedule.

No new model dtype is introduced here. Preserve the Week 2 precision contract
at the `paged_attention` boundary.

## Why Optimize the Paged Path

A conventional attention expression materializes a score matrix with shape
`L × S`. A page-walking implementation can avoid gathering K/V and still make
that intermediate too large. Paged FlashAttention does both:

1. it resolves each K/V tile through `block_table` instead of gathering a dense
   cache;
2. it keeps only a query tile, one K/V tile, and online-softmax state on chip;
3. it writes the normalized output once after all visible pages are consumed.

The algorithm is still exact attention. Only the order of loads and reductions
changes.

## Keep the Day 4 Interface

Do not add a second model-facing operator. Continue to call:

```python
paged_attention(
    query,
    key_pages,
    value_pages,
    block_table,
    context_lens,
    page_size,
    scale=scale,
    mask="causal",
)
```

Put the shape dispatch inside the extension:

| Query shape | Schedule |
|---|---|
| `L <= 8` | Keep the Day 4 vector paged-decode kernel. |
| `L > 8`, BF16, `D == 128` | Use the tiled paged FlashAttention kernel. |

The completed Week 3 model therefore has one paged-attention contract and two
workload-specific GPU schedules.

## Task 1: Tile Queries and Paged K/V

Begin `paged_attention_mma_bf16_d128` in
`src/extensions/src/paged_attention.metal`. Keep
`paged_attention_decode` and `paged_attention_scalar_f32` from Day 4 unchanged;
they remain the short-query and generic controls.

Use eight SIMD groups to cover a 64-row query block. Each SIMD group owns eight
query rows and represents matrix operands as 8×8 fragments. Stage 32 logical
K/V positions per iteration.

For every logical key row in a tile:

```plain
logical_position = tile_start + row
logical_page     = logical_position / page_size
slot             = logical_position % page_size
physical_page    = block_table[batch, logical_page]
address          = pages[physical_page, kv_head, slot, :]
```

Resolve the physical page while staging the tile. The matrix multiply should
not know whether two adjacent logical rows came from adjacent physical pages.

The Qwen path uses 128-token pages and a 32-token K/V tile. An aligned tile is
therefore physically contiguous even when the logical sequence as a whole is
not. Assign each thread contiguous elements through a cooperative block loader
so adjacent lanes issue coalesced reads. Keep a generic loader for a tile that
crosses a page boundary. Reuse the course-owned `CooperativeTileLoader` and
direct `simdgroup_matrix` fragments from Week 2; do not import a Steel loader
or matrix helper. Your solution owns page translation, the contiguous and
cross-page load paths, tile schedule, causal mask, online softmax, primitive,
and dispatch. It does not instantiate MLX attention.

Tail cases are required. A query block, K/V tile, final page, or context may be
partially full, and physical page ids need not be consecutive.

## Task 2: Compute Tiled Online Softmax

Continue modifying `paged_attention_mma_bf16_d128` in
`src/extensions/src/paged_attention.metal`. This task fills the tiled
online-softmax body; it does not add another public function.

For each query tile, maintain one running maximum, one running sum, and an
unnormalized output accumulator per row. For each K/V tile:

1. compute `Q @ Kᵀ` with the Week 2 SIMD-matrix fragments;
2. apply scale and causal bounds;
3. merge the tile maximum into the running maximum;
4. rescale the previous sum and output accumulator;
5. compute exponentials for the current scores and update the running sum;
6. multiply the tile probabilities by V and update the output accumulator.

After the final visible tile, divide each output row by its running sum and
store it using the model-facing dtype.

Multiply the attention scale by `log2(e)` once and use `fast::exp2` for
online-softmax rescaling inside the hot tile loop. This is mathematically
equivalent to natural exponentials and avoids repeating a base conversion.

The causal offset is `context_len - L`. A key at logical position `s` is visible
to query row `l` when:

```plain
s <= l + context_len - L
```

Skip a whole K/V tile when its first key is beyond the last visible key for the
query block. This is both a correctness rule and an important causal-prefill
optimization.

## Task 3: Validate the Page Boundary

Complete the long-query selection in `PagedAttention::eval_gpu` in
`src/extensions/src/paged_attention.cpp`, then test
`paged_attention_mma_bf16_d128` against the Day 4 kernels. Keep
`tiny_llm_ext::paged_attention` and the Python `paged_attention` signature
unchanged.

Use the GPU-debugging ladder from Week 2 Day 3:

1. compare Day 4 page-walking attention with the readable equation written
   with `mlx.core`;
2. compare paged FlashAttention with the Day 4 path;
3. only then benchmark the tiled kernel.

Required fixtures include:

- a context contained in one page;
- a tile that crosses a page boundary;
- non-consecutive physical page ids;
- `L = 65` and a context whose length is not a tile multiple;
- causal decode after the paged prefill;
- GQA where multiple query heads map to one K/V head;
- output dtype remains BF16.

Force `mx.eval` immediately after each operator so compilation, dispatch, and
addressing failures are reported at the responsible call.

```bash
pdm run test --week 3 --day 5
```

## Task 4: Integrate and Measure

Verify the existing dispatch in `Qwen3MultiHeadAttention.__call__` and the
shape selection inside `PagedAttention::eval_gpu`. Task 4 adds no new
extension function.

The Week 3 model should use the tiled paged path automatically for supported
long prefills. Short queries continue through the vector paged-decode schedule.
Neither path gathers a dense K/V tensor. Canonical Week 3 uses MLX quantized
projections, but its cache, paged attention, batching, and scheduling remain
course-owned. This hybrid course path is not the full-MLX baseline.

Measure the completed operator in the continuous-serving trace. Report prompt
range, page size, batch size, hardware, prefill throughput, decode throughput,
request throughput, peak KV storage, and logical KV copy volume:

```bash
pdm run bench-serving-progression --offline --repeats 4 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 --prefill-step 128 \
  --warmup 1 --cooldown-seconds 1 \
  --json-output benchmark_results/task367-final-main/raw/week3-serving-final-main.json
```

FlashAttention is expected to matter more as prefill grows. It should not
replace the Day 4 decode schedule: a one-token query has no query-tile reuse.

On the checked M4 Pro trace, all three course rows share the same projection
seam:

| Storage / attention path | Prefill tok/s | Output tok/s | Decode tok/s | Requests/s | Peak KV | Avoidable KV copy |
|---|---:|---:|---:|---:|---:|---:|
| Dense growth and reconstruction | 711.18 | 35.23 | 57.59 | 0.469 | 1,096 MiB | 209,532 MiB |
| Paged storage + dense gather | 725.46 | 41.64 | 78.53 | 0.555 | not a total peak | 103,445 MiB |
| Direct paged attention | 672.68 | 46.36 | 105.01 | 0.618 | 576 MiB | 504 MiB |

Relative to dense serving, direct paging is 5.4% lower on prefill, 31.6%
higher on output/request throughput, 82.3% higher on decode, 47.4% lower on
measured peak KV storage, and 99.76% lower on avoidable logical copy volume.
The compatibility row's page-pool counter excludes its temporary dense staging
allocation, so it is not a total peak. These are cumulative Week 3 system
results; they do not isolate the Day 5 prefill schedule from paging, direct
decode, allocation, or scheduling, and they do not credit the MLX projection
seam to paged attention.

Use a separate 8K static sweep as a kernel diagnostic after the serving trace.
It shows when query tiling begins to offset page-table overhead, but it does not
measure request turnover, page reuse, or capacity. The
[performance appendix](./appendix-performance.md) records the matched serving
and long-context measurements. Long-context decode remains a Day 4 vector
kernel workload; do not credit a prefill schedule with a decode gain.

```bash
pdm run bench-course-progression --offline --suite course \
  --variant week2 --variant week3 --variant mlx --model qwen3-4b \
  --input-len 8192 --output-len 2 --prefill-logits last \
  --warmup 1 --repeats 4 --cooldown-seconds 1 \
  --json-output benchmark_results/task367-final-main/raw/week3-8k-final-main.json
```

| 8K static checkpoint | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Week 2 course-owned projections | 323.96 | 17.73 |
| Week 3 seam + course paged path | 463.69 | 27.42 |
| Full MLX | 639.73 | 28.37 |

The Week 3 prefill path is 43.1% faster than Week 2 and reaches 72.5% of full
MLX at this shape. This remains a static diagnostic: it does not measure
request turnover, page reuse, admission capacity, or the projection seam
causally. Its decode row is the Day 4 vector schedule, not evidence for the
tiled prefill kernel. Full method and raw samples are in
`benchmark_results/task367-final-main/task367-final-main-benchmark-ledger.md`.

{{#include copyright.md}}
