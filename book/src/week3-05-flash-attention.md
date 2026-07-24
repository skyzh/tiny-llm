# 🚧 Week 3 Day 5: Paged FlashAttention

> 🚧 This chapter is under review and may change.

Day 4 made attention understand paged KV storage. Its first job was correctness:
translate logical token positions through `block_table`, visit every visible
page, and merge the result with online softmax. Today we keep that exact API and
storage layout but replace the long-prefill schedule with tiled
FlashAttention.

This is a required chapter. FlashAttention belongs here rather than in Week 2
because the serving model's real K/V source is now the page pool. Building a
dense-only kernel first would create a second attention path and then require
students to relearn its memory schedule around page translation.

## Prerequisites

This chapter combines ideas already introduced earlier:

- Week 2 Day 4 introduced the online-softmax recurrence.
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
not. Give each thread contiguous elements through a cooperative block loader;
do not repeat the scalar/strided-load bug diagnosed in Week 2 Day 6. Retain a
generic loader for a tile that can cross a page boundary. The reference uses
MLX's low-level Steel block-loader header for this load primitive, but owns the
page translation, tile schedule, online softmax, primitive, and dispatch. It
does not instantiate MLX attention.

Tail cases are required. A query block, K/V tile, final page, or context may be
partially full, and physical page ids need not be consecutive.

## Task 2: Compute Tiled Online Softmax

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

Following the measured MLX kernel, multiply the attention scale by
`log2(e)` once and use `fast::exp2` for online-softmax rescaling inside the hot
tile loop. This is mathematically equivalent to natural exponentials and avoids
repeating a more expensive base conversion.

The causal offset is `context_len - L`. A key at logical position `s` is visible
to query row `l` when:

```plain
s <= l + context_len - L
```

Skip a whole K/V tile when its first key is beyond the last visible key for the
query block. This is both a correctness rule and an important causal-prefill
optimization.

## Task 3: Validate the Page Boundary

Use the GPU-debugging ladder from Week 2 Day 2:

1. compare Day 4 page-walking attention with the readable MLX equation;
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

The Week 3 model should use the tiled paged path automatically for supported
long prefills. Short queries continue through the vector paged-decode schedule.
Neither path gathers a dense K/V tensor.

Measure long prompts as well as the short course checkpoint. Report prompt
length, page size, batch size, hardware, and both prefill and decode throughput:

```bash
pdm run bench --solution tiny_llm --loader week3 --batch-decode \
  --num-seqs 16 --batch-size 4 --prefill-step 128 \
  --min-input-len 512 --max-input-len 512
```

FlashAttention is expected to matter more as prefill grows. It should not
replace the Day 4 decode schedule: a one-token query has no query-tile reuse.

On the reference Qwen3-4B 8K runs, base-2 softmax plus cooperative
contiguous-page loads raised course prefill from the earlier 384.88 tok/s
baseline to a 427.01 tok/s paired median. MLX measured 568.74 tok/s in the
same alternating campaign, so the paged course path reached 75.1%. Long-context
decode remains a separate Day 4 vector-kernel bottleneck; do not credit this
prefill optimization with a decode gain.

The serving performance lab next varies page size, chunk size, batch size, and
request mix without changing this completed operator contract.

{{#include copyright.md}}
