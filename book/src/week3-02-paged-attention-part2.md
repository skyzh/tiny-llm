# Week 3 Day 2: Paged Attention, Part 2

In this chapter, we move from **paged KV storage** to the runtime metadata and execution path needed for **real paged attention**.

Part 1 introduced fixed-size pages, a model-owned page pool shared by layer caches, and per-layer page metadata. That change already improves the storage abstraction, but it does not yet remove the dense gather before attention. To get the full benefit, the attention path itself must understand how to read from pages.

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
slot_mapping: [B] or [num_new_tokens]
```

For the current layer being executed:

- `block_table[b, i]` gives the page id for request `b`'s current-layer logical page `i`
- `context_lens[b]` gives the valid token count for request `b`
- `slot_mapping` tells us where newly generated K/V should be written

This is the bridge between the scheduler and the attention kernel.

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
    scale=None,
    mask="causal",
)
```

With shapes like:

```plain
query:        B, H_q, L, D
key_pages:    P, H_kv, page_size, D
value_pages:  P, H_kv, page_size, D
block_table:  B, max_pages
context_lens: B
```

Compared with the Week 2 dense path, the important difference is that the source length is no longer represented as one contiguous tensor dimension. It is reconstructed logically from the page table.

## Prefill Metadata

During prefill, a chunk may span multiple pages. The runtime needs to know:

- which current-layer pages already existed,
- which new pages were allocated,
- how many valid tokens are in the tail page,
- how to map incoming K/V rows into page storage.

This is why a paged design usually carries a write-side structure such as `slot_mapping`.

## Decode Metadata

During decode, each active request typically writes one token.

The runtime should be able to:

1. compute the destination slot for that token,
2. write K/V into the correct page slot,
3. update the current layer cache's `context_len`,
4. run attention over the full logical context using `block_table`

This is the point where decode stops paying the repeated dense-repack cost from Week 2.

## How This Maps to `tiny-llm`

## `src/tiny_llm/attention.py`

Add a new function:

```python
def paged_attention(...):
    ...
```

The easiest rollout is:

1. first implement it as a gather-then-call wrapper around existing attention,
2. later replace that wrapper with a real paged kernel or paged FlashAttention path.

That preserves correctness while the runtime contracts stabilize.

## `src/tiny_llm/qwen2_week3.py`

The attention module should be able to branch on cache capability:

```python
if cache.supports_paged_attention:
    x = paged_attention(...)
else:
    x = scaled_dot_product_attention_grouped(...)
```

This keeps the model code readable while letting the cache and kernel evolve independently.

## `src/tiny_llm/batch.py`

The scheduler now needs to prepare runtime metadata instead of only dense K/V:

- per-layer page tables for each active request
- padded batch `block_table`
- `context_lens`
- write positions for prefill and decode

This is where continuous batching and paged attention finally connect. In Week 2, batching worked by repacking tensors. In Week 3, batching should work by reusing page tables and updating only the new slots.

## Recommended Incremental Rollout

The safest implementation order is:

1. paged storage
2. dense gather compatibility path
3. `block_table` / `context_lens` plumbing
4. real paged attention dispatch

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
src/tiny_llm/kv_cache.py
src/tiny_llm/batch.py
```

Extend the batch cache and scheduler so they can prepare:

- `block_table`
- `context_lens`
- write-slot metadata

for all active requests.

## Task 2: Define `paged_attention`

```
src/tiny_llm/attention.py
```

Add a paged attention interface whose inputs come from the paged runtime rather than a dense reconstructed `S` dimension.

## Task 3: Dispatch from the Model

```
src/tiny_llm/qwen2_week3.py
```

Update the model so it can route to paged attention when the cache provides paged runtime metadata.

## Task 4: Connect It to Continuous Batching

```
src/tiny_llm/batch.py
```

Update request admission, slot reuse, and request removal so that:

- finished requests free their pages,
- in this teaching implementation, that means freeing pages from every layer cache,
- new requests allocate from the shared pool,
- active decode steps reuse page metadata instead of rebuilding dense K/V.

After this chapter, the serving stack has the right structure for a real high-throughput runtime: paging is no longer just a storage trick, but part of the execution model itself.

{{#include copyright.md}}
