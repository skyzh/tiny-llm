# 🚧 Week 3 Day 5: Paged Attention, Part 2

> 🚧 This chapter is under review and may change.

In this chapter, we move from **paged KV storage** to the runtime metadata and execution path needed for **real paged attention**.

Part 1 introduced fixed-size pages, one model-owned physical pool per layer,
and request-local page metadata. That change already improves the storage
abstraction, but it does not yet remove the dense gather before attention. To
get the full benefit, the attention path itself must understand how to read
from pages.

> **Prerequisite:** Complete Day 2 FlashAttention before this chapter. Paged
> prefill reuses its BF16 SIMD-matrix tiles and online-softmax recurrence; the
> new concept here is translating logical K/V positions through a block table.

## Paged KV Cache vs Paged Attention

These two ideas are related, but they are not the same:

1. **Paged KV cache**
   KV is stored in fixed-size pages.
2. **Paged attention**
   The attention path reads KV directly from those pages via metadata such as a page table.

You can implement the first one without the second one, but the real serving payoff comes when both are present.

## The Metadata a Paged Runtime Needs

Once KV is paged, dense `B x H x S x D` tensors are no longer the natural runtime representation. Instead, the runtime should prepare metadata like:

```plain
block_table:  [B, max_pages_per_request]
context_lens: [B]
```

For the current layer being executed:

- `block_table[b, i]` gives the page id for request `b`'s current-layer logical page `i`
- `context_lens[b]` gives the valid token count for request `b`

This is the bridge between the scheduler and the attention kernel.

A production runtime often also carries write-side metadata such as `slot_mapping`.
For this chapter, we keep the write side inside the cache and focus on the read-side
metadata needed by attention.

## Why `block_table` Matters

Suppose one layer cache for request A has:

```plain
page_ids = [12, 5, 3]
context_len = 10
page_size = 4
```

Then the logical sequence positions map to physical storage like this:

```plain
logical 0..3  -> page 12
logical 4..7  -> page 5
logical 8..9  -> page 3
```

The attention runtime does not need a fully gathered dense tensor if it already knows:

- which current-layer page each logical block lives in,
- how long the context is,
- and where the current query positions are.

That is exactly what `block_table` and `context_lens` encode.

## The Real Attention API

At this point, the runtime should grow a new attention entry point:

```python
paged_attention(
    query,
    key_pages,
    value_pages,
    block_table,
    context_lens,
    page_size,
    scale=None,
    mask="causal",
)
```

With shapes like:

```plain
query:          B, H_q, L, D
key_pages[i]:   1, H_kv, page_size, D
value_pages[i]: 1, H_kv, page_size, D
block_table:    B, max_pages
context_lens:   B
```

Compared with the earlier dense path, the important difference is that the source length is no longer represented as one contiguous tensor dimension. It is reconstructed logically from the page table.

In this chapter, `paged_attention` should read pages directly from a GPU
kernel. The runtime contract is now: model code and batching code pass pages
plus metadata, and the attention kernel walks that metadata without first
rebuilding dense K/V.

## Prefill Metadata

During prefill, a chunk may span multiple pages. The runtime needs to know:

- which current-layer pages already existed,
- which new pages were allocated,
- how many valid tokens are in the tail page,
- how to map incoming K/V rows into page storage.

In this teaching implementation, the cache still owns the write-side bookkeeping.
The attention path only needs the block table after the write is done.

## Decode Metadata

During decode, each active request typically writes one token.

The runtime should be able to:

1. append the token's K/V to the current tail page,
2. allocate a new page only if the tail page is full,
3. update the current layer cache's `context_len`,
4. run attention over the full logical context using `block_table`

This is the point where decode stops paying the repeated dense-repack cost from Day 1.

## Choose a Schedule for Each Query Shape

Before implementing the GPU path, separate decode from prefill. A single tile
shape cannot keep the GPU busy for both a one-token query and a long prompt.
Use these design rules:

1. Preserve BF16 Q/K/V and output storage. Accumulate dot products,
   online-softmax statistics, and output fragments in FP32.
2. For short queries, expose parallelism across the cached context. Do not
   reserve most of a threadgroup for query rows that do not exist.
3. For prefill, expose parallelism across query rows and use SIMD-matrix
   fragments for the dense work inside each page tile.
4. Keep scalar FP32 kernels as correctness oracles, not as the default GPU
   schedule.

Start with this dispatch plan and treat its thresholds as values to verify on
your hardware:

| Shape | Course-owned dispatch | Work decomposition |
|---|---|---|
| `L <= 8`, FP32 or BF16 | Vector paged decode | One threadgroup per query row; 32 SIMD groups stride over the context and merge partial `(max, sum, output)` states. |
| `L > 8`, BF16, `D == 128` | SIMD-matrix paged prefill | Eight SIMD groups cover a 64-row query block, stage 32 K/V rows at a time through the block table, and use 8×8 matrix fragments. |
| `L > 8`, FP32 | Scalar paged prefill | A small correctness/reference path with 16 query rows and 32 K/V rows per tile. |
| CPU FP32 | Scalar reference | A readable oracle used for correctness rather than the performance target. |

Put the shape decision at the extension boundary rather than converting inputs
or falling back to dense attention in Python. Benchmark values immediately
below and above each threshold while keeping the model-facing
`paged_attention` API unchanged.

## How This Maps to `tiny-llm`

### `src/tiny_llm/attention.py`

Add a new function:

```python
def paged_attention(...):
    ...
```

For this course implementation, make it a FlashAttention-style page-walking
Metal kernel:

1. use `block_table[b]` to find the physical pages for request `b`,
2. use `context_lens[b]` to ignore unused tail capacity,
3. visit K/V in small tiles instead of materializing dense K/V,
4. merge each tile into the output with online softmax.

The important change from dense attention is the K/V address calculation. Dense attention can
advance through dense K/V by pointer arithmetic. Week 3 must translate each
logical key position through `block_table` first:

```plain
logical key position -> logical page -> physical page id -> slot in page
```

After that lookup, the online-softmax update is the same idea as Day 2
FlashAttention. For BF16 prefill, reuse the Day 2 SIMD-matrix tile directly and
replace only its dense K/V loads with page-table loads. We still avoid a dense
K/V gather before attention.

One-token decode needs a different work decomposition. A 64-row prefill tile
would leave almost every query row idle, so dispatch short queries to a
vector-oriented kernel that partitions the context across SIMD groups and
merges their partial online-softmax states. Do not run decode through a fixed
32-row scalar prefill tile.

The page pool should therefore expose contiguous physical storage:

```plain
key_pages:   P, H_kv, page_size, D
value_pages: P, H_kv, page_size, D
```

A Python list of page tensors is convenient for teaching the allocator, but a
GPU kernel needs a single buffer so `page_id` can be turned into an address.

### `src/tiny_llm/qwen3_week3.py`

The attention module should call the paged runtime directly:

```python
metadata = cache.update_and_fetch_paged(...)
x = paged_attention(...)
```

Week 3 cache handles are expected to provide paged metadata. If a dense cache is
passed to the Week 3 model, that is a programming error rather than a signal to
silently fall back to dense attention.

### `src/tiny_llm/batch.py`

The scheduler now needs to prepare runtime metadata instead of only dense K/V:

- per-layer page tables for each active request
- padded batch `block_table`
- `context_lens`

This is where continuous batching and paged attention finally connect. On Day 1, batching worked by repacking tensors. Here, batching should work by reusing page tables and updating only the new slots.

## Recommended Incremental Rollout

The safest implementation order is:

1. paged storage
2. `block_table` / `context_lens` plumbing
3. FlashAttention-style page-walking GPU attention
4. model and batch dispatch

This order matters because it gives us a clean correctness baseline at each step.

## Correctness Invariants

These are the invariants worth checking in tests:

1. `context_len` always equals the number of written logical token positions.
2. `block_table` reconstructs the same logical KV order as the dense baseline.
3. the allocator never hands the same page to two live cache handles unless explicit sharing is implemented.
4. releasing a request returns all pages owned by all of its layer caches exactly once.
5. decode allocates a new page only when the tail page overflows.

## Task 1: Add Batch Metadata

```
src/tiny_llm/paged_kv_cache.py
src/tiny_llm/kv_cache.py
src/tiny_llm/batch.py
```

Extend the batch cache and scheduler so they can prepare:

- `block_table`
- `context_lens`

for all active requests.

## Task 2: Define `paged_attention`

```
src/tiny_llm/attention.py
src/extensions/src/paged_attention.cpp
src/extensions/src/paged_attention.metal
```

Add a paged attention interface whose inputs come from the paged runtime rather than a dense reconstructed `S` dimension. Preserve BF16 Q/K/V and output; scores, softmax statistics, and output accumulators remain FP32.

The reference solution walks every request's block table and keeps online
softmax state:

```python
running_max = max(previous_max, page_max)
running_sum = previous_sum * exp(previous_max - running_max) + page_sum
output = previous_output * exp(previous_max - running_max) + page_output
```

After all visible pages are consumed, divide `output` by `running_sum`.
This is the key idea that lets the kernel avoid materializing dense K/V while
still producing the same result as dense attention.

Implement two GPU dispatches:

1. For `L <= 8`, partition logical context positions across SIMD groups and
   merge their partial `(max, sum, output)` states in threadgroup memory. The
   reference schedule uses 32 SIMD groups per query so group `g` visits
   positions `g`, `g + 32`, `g + 64`, and so on.
2. For BF16 `D == 128` prefill, start from Day 2's SIMD-matrix FlashAttention
   kernel and resolve each staged K/V row through `block_table`. Use a 64-row
   query block and 32-row K/V tiles so all eight SIMD groups own useful query
   rows.

Keep a scalar FP32 path for small correctness fixtures and the CPU reference.

### Course implementation boundary

MLX remains the array runtime for shapes, reshapes, transposes, contiguous
storage, dtype conversion, allocation, and custom-primitive dispatch. The
attention implementation itself must remain course-owned: do not call
`mx.fast.scaled_dot_product_attention`, reuse an MLX attention/Steel kernel, or
reconstruct dense K/V and express the paged operator as MLX matmul plus
softmax. MLX SDPA may appear only in tests and benchmarks as an external
correctness/performance reference.

Ordinary prefill runs the course-owned Day 2 FlashAttention kernel directly on
the freshly projected dense K/V, then stores those same tensors in the paged
cache for decode. It must not gather them back from the cache. Paged prefill
remains useful when chunked prefill already has history in the page pool.

## Task 3: Dispatch from the Model

```
src/tiny_llm/qwen3_week3.py
```

Update the model so it can route to paged attention when the cache provides paged runtime metadata.

FlashAttention prefill and the paged paths must share one request cache. For an
ordinary first prefill with more than eight tokens, append BF16 K/V to the
paged cache and run the Day 2 FlashAttention kernel directly on the freshly
projected K/V. Do not gather the tensors back from the cache. If chunked
prefill already has cached history, pass page metadata to the paged BF16
prefill kernel instead. Subsequent short decode steps always use the
vector-oriented paged kernel. Neither path may upcast the complete Q/K/V
tensors or change cache dtype. Keep this dispatch behind the Week 3 model so
Week 2 remains independent. The final Week 3 model selects FlashAttention
prefill automatically for supported `D == 128` models; `--disable-flash-attn`
exists only for the measured paged-prefill ablation.

This creates the final routing policy:

```plain
fresh prefill, L > 8, D == 128 -> Day 2 dense FlashAttention on fresh K/V
long prefill with cached history -> paged BF16 SIMD-matrix attention
decode or another short chunk    -> paged vector attention
```

All three paths write or retain the same paged cache. The first path is
"zero-gather" because it stores fresh K/V in the cache but attends to the
already-available projection tensors instead of gathering an equivalent dense
copy back out. `--disable-paged-attention` is a Day 4 dense-gather teaching
ablation, not the completed Day 5 serving path.

## Task 4: Connect It to Continuous Batching

```
src/tiny_llm/batch.py
```

Update request admission, slot reuse, and request removal so that:

- finished requests free their pages,
- in this teaching implementation, that means freeing pages from every layer cache,
- new requests allocate from the corresponding layer pool,
- active decode steps reuse page metadata instead of rebuilding dense K/V.

After this chapter, the serving stack has the right structure for a real high-throughput runtime: paging is no longer just a storage trick, but part of the execution model itself.

## Performance Lab: Make Paging Earn Its Cost

The goal of this lab is to decide when direct page traversal is useful. Paged
attention is not automatically a faster attention operator: it trades regular,
contiguous K/V access for flexible allocation and removes the dense repack that
would otherwise happen before attention. Your measurements must include both
sides of that trade.

Start by recording three operator baselines on the same machine:

1. the dense course attention path, including any required K/V gather,
2. your direct paged-attention path,
3. the MLX attention path as a production-library reference.

These M4 Pro measurements are reference points, not target values to copy into
your report:

| Batch / context | Dense course attention | Course paged attention | MLX attention |
|---|---:|---:|---:|
| 1 / 128 | 226 µs | 219 µs | 170 µs |
| 1 / 512 | 300 µs | 309 µs | 192 µs |
| 1 / 2048 | 729 µs | 726 µs | 263 µs |
| 8 / 512 | 800 µs | 626 µs | 225 µs |
| 8 / 2048 | 1,347 µs | 1,322 µs | 460 µs |

### Checkpoint 1: Establish a Correct Direct Path

Implement the simplest page-walking kernel first. Verify that it:

- reads K/V through `block_table` without constructing a dense K/V tensor,
- ignores unused slots in the final page,
- matches dense attention for several page boundaries and context lengths,
- supports grouped-query attention when `H_q != H_kv`.

A correct first version may be slower than dense attention. At this checkpoint,
the useful result is a trustworthy baseline and a working runtime interface.

### Checkpoint 2: Design the Decode Schedule

Optimize for the one-token decode shape instead of treating page traversal as a
serial loop. Work through these changes one at a time and benchmark after each
one:

1. assign the lanes of a SIMD group to adjacent elements of a head so K/V loads
   can be coalesced,
2. load a page-table entry once and reuse it for all positions in that page,
3. keep the query and online-softmax state in registers across page tiles,
4. combine partial dot products with SIMD reductions instead of threadgroup
   scratch memory and repeated barriers,
5. specialize the `L = 1` decode case so it does not carry prefill control flow,
6. prepare the batch's page metadata once per scheduler step rather than once
   per layer or attention head.

For each change, explain which cost it targets: memory traffic, synchronization,
address calculation, or dispatch overhead. Keep a change only when the measured
result supports the explanation.

### Checkpoint 3: Evaluate the Serving System

Operator latency alone does not capture the purpose of paging. Run an
end-to-end workload with requests entering and leaving the batch, then report:

- time per decode step and aggregate tokens per second,
- peak KV-cache memory and the number of live requests admitted,
- bytes or time spent gathering and repacking K/V,
- paged-attention latency relative to the dense course path and MLX.

On Apple Silicon, retain a dense path for workloads where contiguous attention
is faster. Use paged attention when eliminating repacks, reusing pages across
scheduler steps, or admitting more concurrent requests improves the system-level
result. The lesson is to make dispatch a measured policy decision, not to assume
that paging must win every operator benchmark.

```bash
pdm run test --week 3 --day 5
```

{{#include copyright.md}}
