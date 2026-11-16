# 🚧 Week 3 Day 5: Paged FlashAttention

> 🚧 This chapter is under review and may change.

In this chapter, you will replace the correctness-first implementation
for the supported BF16 long-prefill hot case with paged FlashAttention. The
operator still translates logical K/V positions through `block_table`, but it
now stages page-backed tiles on chip and combines them with online softmax.
Short decode and shapes outside the optimized region keep their correct Day 4
fallbacks.

Day 5 starts differently from an ordinary correctness checkpoint: the public
behavior tests may already pass with your Day 4 implementation. That is
intentional. Your work is to make the optimized path real and reachable while
preserving the same observable behavior. The manual control-flow trace at the
end of the chapter is the completion feedback for that performance change; it
is guidance, not a hidden grading requirement.

This is a required chapter. Week 2's dense tiled kernel teaches the
online-softmax update, but the serving model's K/V source is now the page
pool. Reuse that arithmetic while translating each K/V tile through the
block table; the dense loading schedule cannot read paged state directly.

## Prerequisites

This chapter combines four prerequisites:

- [Week 2 Day 5](./week2-05-tiled-prefill-attention.md) introduced the
  online-softmax recurrence for tiled dense prefill attention.
- [Week 2 Day 3](./week2-03-simd-matrix-prefill.md) introduced cooperative
  SIMD-matrix fragments for quantized projection prefill.
- Week 3 Day 3 introduced physical pages and block tables.
- Week 3 Day 4 introduced direct page-walking attention and the decode
  schedule.

Today adds no new model dtype. Keep the same precision as Week 2 at the
`paged_attention` boundary.

## Why Optimize the Paged Path

A conventional attention expression materializes a score matrix with shape
`L × S`. A page-walking implementation can avoid gathering K/V and still make
that intermediate too large. Paged FlashAttention does both:

1. it resolves each K/V tile through `block_table` instead of gathering a dense
   cache;
2. it keeps only a query tile, one K/V tile, and online-softmax state on chip;
3. it writes the normalized output once after all visible pages are consumed.

The algorithm remains numerically equivalent attention under the course's BF16
rounding tolerance. Only the order of loads and reductions changes.

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

The reference solution puts this shape dispatch inside the extension:

| Query shape | Schedule |
|---|---|
| `L <= 8` | Keep the Day 4 vector paged-decode kernel. |
| `L > 8`, BF16, `D == 128` | Use the tiled paged FlashAttention kernel. |
| Every other supported shape | Keep a correct direct page-walking fallback. |

So the finished Week 3 model has one paged-attention interface, with a faster
path for long BF16 prefill. The tests do not check kernel names, the dispatch
threshold, tile shapes, private flags, helpers or file layout. Any
implementation that behaves the same passes, including one built on an
external low-level library.

## Task 1: Tile Queries and Paged K/V

The reference walkthrough begins `paged_attention_mma_bf16_d128` in
`src/extensions/src/paged_attention.metal`. Keep
`paged_attention_decode`, `paged_attention_scalar_f32`, and
`paged_attention_scalar_bf16` from Day 4 unchanged; they remain the short-query
and generic controls.

Use eight SIMD groups to cover a 64-row query block. Each SIMD group owns eight
query rows and represents matrix operands as 8×8 fragments. Stage 32 logical
K/V positions per iteration.

The reference keeps each lane's two fragment elements in ordinary vectors
between matrix operations: `float2` for FP32 scores and output accumulators, and
`vec<bfloat, 2>` for Q, K, V and the rounded probabilities. Build the
`simdgroup_matrix` operands at the multiply-accumulate call, then copy its
two result elements back. This changes how long the matrix objects live,
not the attention equation. Keep the same lane coordinates, row-reduction
order, mask and BF16 conversion points.

How much does it help? On one M4 Pro with Qwen3-4B, with a fixed queue of
eight prompts of 649 to 1,462 tokens prefilled in chunks of up to 256 tokens,
whole-queue throughput increased by 6.89% with one decode slot and 7.34% with
four. Those are medians
over six paired runs each, and every pair favored the new representation. Peak
MLX allocation did not change. The measurement covers the whole queue, not the
time to each request's first token, and it is one machine and one workload.
Measure your own requests before counting it as a gain on yours.

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
crosses a page boundary. The reference reuses the
`CooperativeTileLoader` from Week 2 and the same 8×8 `simdgroup_matrix`
multiply, so those mechanisms remain visible. You may choose a different internal helper or
low-level library as long as the operator retains direct page-table semantics
and equivalent public behavior.

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
3. trace the model-to-kernel route, then design a matched operator benchmark
   before making a speed claim.

The behavior fixtures cover:

- a context contained in one page;
- a partial query block and context tail with non-consecutive physical pages;
- poisoned unused pages and tail slots, which must not affect the result;
- batched GQA where multiple query heads map to one K/V head;
- causal and non-causal calls, including an explicit scale;
- short decode and generic BF16 head-dimension fallbacks;
- a model-level long-prefill comparison through the public Week 3 path;
- output shape and dtype, plus numerical agreement with dense attention.

These are public input/output checks. They do not inspect allocator choices,
page identifiers chosen by an implementation, kernel names, or routing state.

Force `mx.eval` immediately after each operator so compilation, dispatch, and
addressing failures are reported at the responsible call.

The new kernel and dispatch live in the extension, so rebuild it before every
test run; otherwise the tests still exercise your previous Day 4 build:

```bash
pdm run build-ext
pdm run test --week 3 --day 5
```

If this command is green before you begin Day 5, you have confirmed that Day
4's correctness boundary is intact. Continue with the optimized implementation
and use the trace below to confirm that the supported hot case no longer takes
the scalar fallback.

## Task 4: Integrate and Measure

Verify the existing dispatch in `Qwen3MultiHeadAttention.__call__` and the
shape selection inside `PagedAttention::eval_gpu`. Task 4 adds no new
extension function.

The Week 3 model should use the tiled paged path automatically for supported
long prefills. Short queries continue through the vector paged-decode schedule.
Neither path gathers a dense K/V tensor. The Week 3 model sends its
projections to MLX, but the cache, paged attention, batching and scheduling are
yours, so it is not the same as the full-MLX baseline.

The course does not yet have a benchmark that compares today's tiled schedule
with Day 4's kernel on the same long BF16 prompts, changing nothing else. The
runs below measure the whole server, so they cannot show what today's kernel
alone gained.

Run the whole-server check on your solution:

```bash
week3_result_root="$HOME/tiny-llm-week3-results"
mkdir -p "$week3_result_root"
week3_result_dir="$(mktemp -d "$week3_result_root/run-XXXXXX")"

pdm run bench-serving-progression --solution tiny_llm --offline --repeats 4 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 --prefill-step 128 \
  --warmup 1 --cooldown-seconds 1 \
  --json-output "$week3_result_dir/serving-tiny-llm.json"
```

This writes a new JSON file. To compare with the reference solution, rerun the
same flags with `--solution ref` and a different file name.

FlashAttention should matter more as prompts get longer, but treat that as a
hypothesis until a matched benchmark measures it. It does not replace Day 4's
decode kernel: a one-token query has nothing to reuse across a query tile.

For a sense of scale, here is an earlier run of the same command on an M4 Pro,
made before Week 2 was rebuilt. All three rows sent projections to MLX in the
same way:

<div class="table-wrapper">
<table>
  <thead>
    <tr><th scope="col">Storage / attention path</th><th scope="col">Prefill tok/s</th><th scope="col">Output tok/s</th><th scope="col">Decode tok/s</th><th scope="col">Requests/s</th><th scope="col">Peak KV</th><th scope="col">Avoidable KV copy</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">Dense growth and reconstruction</th><td>711.18</td><td>35.23</td><td>57.59</td><td>0.469</td><td>1,096 MiB</td><td>209,532 MiB</td></tr>
    <tr><th scope="row">Paged storage + dense gather</th><td>725.46</td><td>41.64</td><td>78.53</td><td>0.555</td><td>not a total peak</td><td>103,445 MiB</td></tr>
    <tr><th scope="row">Direct paged attention</th><td>672.68</td><td>46.36</td><td>105.01</td><td>0.618</td><td>576 MiB</td><td>504 MiB</td></tr>
  </tbody>
</table>
</div>

Compared with dense serving in that run, direct paging was 5.4% slower on
prefill, 31.6% faster on output and requests, and 82.3% faster on decode. It
used 47.4% less peak K/V storage, and the K/V copying it could have avoided
fell by 99.76%. The middle row's counter leaves out its temporary dense copy, so it is not
a total peak. These are results for the whole server, so they mix paging,
direct decode, allocation and scheduling; none of it is today's prefill kernel
alone.

A second check runs one long 8,192-token prompt through the Week 2 model, the
Week 3 model and MLX. It also measures the whole model, with one request and
no pages being reused, so it says nothing about the scheduler or about
today's kernel alone:

```bash
week3_result_root="$HOME/tiny-llm-week3-results"
mkdir -p "$week3_result_root"
week3_result_dir="$(mktemp -d "$week3_result_root/run-XXXXXX")"

pdm run bench-course-progression --solution tiny_llm --offline --suite course \
  --variant week2 --variant week3 --variant mlx --model qwen3-4b \
  --input-len 8192 --output-len 2 --prefill-logits last \
  --warmup 1 --repeats 4 --cooldown-seconds 1 \
  --json-output "$week3_result_dir/8k-tiny-llm.json"
```

To compare with the reference solution, rerun with `--solution ref` and a new
file name. The earlier run gave the table below. Its Week 2 row came from the
older version of Week 2, so your `week2` row will not match it.

<div class="table-wrapper">
<table>
  <thead>
    <tr><th scope="col">8K static checkpoint</th><th scope="col">Prefill tok/s</th><th scope="col">Decode tok/s</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">Week 2: your projection kernels</th><td>323.96</td><td>17.73</td></tr>
    <tr><th scope="row">Week 3: MLX projections + your paged path</th><td>463.69</td><td>27.42</td></tr>
    <tr><th scope="row">Full MLX</th><td>639.73</td><td>28.37</td></tr>
  </tbody>
</table>
</div>

In that run, Week 3 prefill was 43.1% faster than the older Week 2 and
reached 72.5% of MLX's speed at this length. That older Week 2 is not today's
`selected` checkpoint, so the 43.1% does not carry over. The decode column
runs Day 4's kernel, not today's. The
[performance appendix](./appendix-performance.md) has the full method and raw
samples.

> **Trace the implementation, not its names.** Copy the prompt below into GPT
> or Claude after your behavior tests pass. This is learner guidance and is not
> part of automated grading.
>
> ```text
> Trace the actual control flow for Week 3 long-prefill paged attention. First determine whether the implementation stays in this repository or crosses into an external low-level library. For a repository implementation, start at the Python model call and follow native dispatch into Metal. For an external implementation, trace from the Python model call to the library boundary and record equivalent boundary evidence, including the call site, arguments, and selected backend. Determine whether BF16 queries with L > 8 and D == 128 reach the learner's optimized tiled or FlashAttention-equivalent implementation rather than the correctness fallback. Verify that short decode and unsupported or generic shapes retain correct fallback behavior. Cite file and line evidence for repository code and equivalent boundary evidence for external code, flag dead or unreachable paths, judge control flow rather than function or kernel names, and report findings only—do not modify files.
> ```

{{#include copyright.md}}
