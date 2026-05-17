# Week 3 Day 1: Paged Attention, Part 1

In this chapter, we will design the **paged KV cache**. This is the storage abstraction behind paged attention.

By the end of Week 2, our serving stack already supports:

- per-request KV cache
- chunked prefill
- continuous batching
- FlashAttention

That gives us a working miniature serving engine, but the memory layout is still too simple. KV for each request is treated as one growing dense tensor, and batching rebuilds dense K/V for all active requests. That approach is easy to teach, but it does not scale well once requests become long and numerous.

Paged attention starts by fixing the storage layout.

**📚 Readings**

- [vLLM Paged Attention Design](https://docs.vllm.ai/en/v0.18.0/design/paged_attention/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

## Why the Week 2 KV Layout Becomes Expensive

Right now, the mental model looks like this:

```plain
request A -> one dense KV tensor
request B -> one dense KV tensor
request C -> one dense KV tensor
```

Before attention, the runtime repacks them into:

```plain
keys:   [B, H, S_max, D]
values: [B, H, S_max, D]
mask:   [B, 1, L, S_max]
```

The trouble is that decode only adds a tiny amount of new information each step, but the dense layout keeps revisiting old KV.

For example, if a request already has 17 cached tokens and we decode 1 more token:

```plain
new useful work: append 1 token
dense repack view: rebuild 18 logical positions
```

For one request this is fine. For many live requests, the runtime spends more and more time moving previously computed KV instead of doing actual model work.

## The Page Abstraction

Instead of storing each layer's KV for a request as one long tensor, we divide storage into fixed-size **pages**:

```plain
key_pages:   pages with up to page_size token slots
value_pages: pages with up to page_size token slots
```

Each layer cache keeps a small page table:

```plain
page_ids = [12, 5, 3]
context_len = 10
```

That means:

```plain
page 12 -> tokens 0..3
page  5 -> tokens 4..7
page  3 -> tokens 8..9
```

The logical sequence is still length 10. The difference is that the runtime is no longer forced to represent it as one contiguous tensor.

In our Day 1 teaching implementation, those fixed-size pages live in one shared **page pool** owned by the model. Every layer cache receives that same pool, but each layer cache keeps its own `page_ids`, `page_lens`, and `offset`.

In the reference solution, `page_size` is the physical page capacity. Unused tail slots are not part of the logical sequence; `page_lens` decides which prefix of each page is valid.

## Why Fixed-Size Pages Help

The page abstraction gives us two immediate wins:

1. Appending a token usually updates only the current tail page in the pool.
2. Finished requests can return their pages to a shared free list.

This is the key memory-management idea behind paged attention systems such as vLLM.

## Data Structures We Need

## 1. `PagePool`

The model should own one pool with a model-wide page allocator and flat K/V page storage:

```plain
free_pages: available page ids for the whole model
keys[page_id]:   physical key page
values[page_id]: physical value page
```

Each layer still has distinct K/V contents because each layer cache allocates its own physical pages. In this teaching version, each layer cache also has its own logical page table. That is simpler than nano-vllm's shared block table: layer 0 might own pages `[0, 1]`, while layer 1 owns pages `[2, 3]`, but both page sets came from the same model-owned pool.

In the reference solution, this becomes `TinyKvPagedPool`.

## 2. `PagedRequestCache`

A layer cache for one request should track:

- `page_ids`
- `page_lens`
- `offset`
- `page_size`

Derived values:

- `num_pages = len(page_ids)`
- `context_len = offset`
- `last_page_fill = page_lens[-1]` when at least one page exists

In the reference solution, this becomes `TinyKvPagedCache`.
It is created with a pool from the model. It should not allocate its own pool,
because that would isolate one request from the shared page allocator.

The reference solution creates one `TinyKvPagedCache` per transformer layer. Those caches share the pool, but they do not share metadata: each layer cache owns its own `page_ids`, `page_lens`, and `offset`.

## 3. Tail-Append Logic

When new K/V arrives for one layer:

1. look at that layer cache's last page
2. if there is room, append only the new slice into the tail page
3. otherwise allocate a new page and continue writing
4. update cache metadata such as `page_lens` and `offset`

This replaces the Week 2 pattern of repeatedly concatenating along the sequence dimension.

## Prefill with Pages

Suppose `page_size = 4` and one prefill chunk contains 6 tokens:

```plain
chunk = [t0 t1 t2 t3 t4 t5]
```

One possible layout is:

```plain
page 7 <- [t0 t1 t2 t3]
page 2 <- [t4 t5]        # 2 valid tokens, 2 unused slots of capacity
```

That layer cache's metadata becomes:

```plain
page_ids = [7, 2]
context_len = 6
```

The important property is that a later decode token can be appended to page `2` without touching page `7`.

## Decode with Pages

During decode, each live request adds one token at a time.

With paged storage:

1. compute one-token `k` and `v`
2. check whether the tail page still has space
3. write into that page if possible
4. allocate a new page only when the old one is full

So if `page_size = 4` and `context_len = 9`:

```plain
page_ids = [12, 5, 3]
```

Appending token 9 only updates the last page instead of rebuilding all earlier KV.

## Stage A: Keep Dense Attention

The cleanest first implementation is **paged storage with dense gather**.

That means:

- pages in the shared pool are the source of truth,
- layer caches stop owning one monolithic K/V tensor,
- layer caches only track page metadata,
- attention still receives dense K/V reconstructed from pages.

This is not the final paged attention runtime yet, but it is a very useful intermediate step:

- small surface-area change
- easier debugging
- direct correctness comparison against `TinyKvFullCache`

## How This Maps to `tiny-llm`

## `src/tiny_llm/paged_kv_cache.py`

Add:

- `TinyKvPagedPool`
- `TinyKvPagedCache`

Keep `TinyKvFullCache` in `src/tiny_llm/kv_cache.py` as a baseline and test
oracle.

The key Day 1 behavior is:

1. write new K/V into the layer cache's tail page or newly allocated pages,
2. gather the layer cache's pages back into dense K/V,
3. feed that dense K/V into the old attention path.

So Day 1 changes the storage model first, not the attention kernel yet.

## `src/tiny_llm/batch.py`

Requests should own per-layer cache handles instead of long dense K/V tensors.

The scheduler should still:

- perform chunked prefill,
- hold active requests,
- free cache pages when a slot finishes.

The difference is that freeing a request now means releasing all pages owned by its layer caches back to the pool.

Day 1 also keeps a small `rewind(n)` lifecycle hook. Rewind is useful for speculative decoding: if some drafted tokens are rejected, the cache must forget their K/V. In the paged cache, rewind frees whole pages that are no longer needed and shortens the valid length of the final remaining page.

## Design Questions for Day 1

Before implementing, make sure the following are clear:

1. What page size should this repo use for teaching?
2. How do we represent the free-page allocator?
3. How do we prove that paged storage reconstructs the same logical KV as `TinyKvFullCache`?
4. How do layer cache handles share one pool while keeping their own page metadata?
5. When do we materialize page writes to avoid MLX lazy-graph growth?

## Task 1: Design `PagePool`

```
src/tiny_llm/paged_kv_cache.py
```

Design a model-owned page pool that:

- owns the model-wide free-page allocator,
- stores flat fixed-size K/V pages,
- allocates and frees page ids,
- supports writing a chunk into page storage,
- is shared by every layer cache created by the model.

## Task 2: Design `PagedRequestCache`

```
src/tiny_llm/paged_kv_cache.py
```

Replace the "one layer cache = one dense KV tensor" model with:

- `page_ids`
- `context_len`
- append logic over fixed-size pages
- `release()` for returning pages on request completion
- `rewind(n)` for dropping the newest `n` logical tokens

## Task 3: Add a Dense-Gather Compatibility Path

```
src/tiny_llm/paged_kv_cache.py
src/tiny_llm/qwen3_week3.py
```

Build a compatibility path that reconstructs dense K/V from pages and compares it against `TinyKvFullCache`.

This gives us a correctness checkpoint before we change the attention path itself.

In the next chapter, we will take the next step: instead of gathering dense K/V before attention, we will pass runtime metadata such as `block_table` directly into a paged attention path.

{{#include copyright.md}}
