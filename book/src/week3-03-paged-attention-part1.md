# 🚧 Week 3 Day 3: Paged KV Cache

> 🚧 This chapter is under review and may change.

After Day 2, your server keeps several requests in flight. Each one has its own
K/V cache for every layer, stored as one dense tensor, and requests start,
grow, and finish at different times. At every decode step,
`BatchingKvCache` copies each request's cached keys and values into one batch
tensor shaped for the longest request:

```plain
keys:   [B, H, S_max, D]
values: [B, H, S_max, D]
mask:   [B, 1, L, S_max]
```

Open the `BatchingKvCache.update_and_fetch` you wrote on Day 1 and you can see
it happen: the method allocates fresh `keys` and `values` tensors and copies
every active request's cache into them. Decode adds one token per request, but
this step copies the full history of every request. A request with 500 cached
tokens has 501 positions copied to produce one new token. (The reference
solution counts these bytes, and the batch benchmark prints them as
`Dense KV bytes copied into batch tensors`.)

The storage itself is awkward too. A dense cache either reserves room for
the longest sequence a request might reach, which wastes memory on requests
that finish early, or grows as it goes, which means copying everything it has
stored so far.

Operating systems solved the same problem for process memory long ago: hand
out memory in small fixed-size pages, and keep a table that says where each
page lives. Today you do the same for K/V. You will:

1. store K/V in fixed-size **pages** inside one shared pool per layer;
2. give every request a small **page table** that records which pages hold its
   tokens, in order;
3. append new tokens by writing only into the request's last page, or a fresh
   page when the last one is full; and
4. return a finished request's pages to the pool so the next request can reuse
   them.

Attention does not change yet. In this chapter, each layer cache copies its
pages back into an ordinary dense tensor before attention runs, so you can
check the new storage against the old cache token for token. Day 4 teaches
attention to read the pages directly and removes that copy.

**📚 Readings**

- [vLLM Paged Attention Design](https://docs.vllm.ai/en/v0.18.0/design/paged_attention/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

## Pages by Example

Use a page size of 4 tokens to keep the numbers small. (The model uses 128 by
default. The tests use 4.) Everything below happens inside the pool for one
transformer layer.

**Request A arrives with a 6-token prompt.** The pool is empty, so it hands out
pages 0 and 1:

```plain
page 0: [a0 a1 a2 a3]
page 1: [a4 a5 __ __]       2 of 4 slots used

A: page_ids = [0, 1]   page_lens = [4, 2]   offset = 6
```

`page_ids` lists A's pages in logical order. `page_lens` says how many slots of
each page hold real tokens. `offset` is the number of tokens A has cached.

**Request B arrives with a 3-token prompt.** It gets the next new page:

```plain
page 2: [b0 b1 b2 __]

B: page_ids = [2]   page_lens = [3]   offset = 3
```

**A decodes three tokens.** The first two fit in the empty slots of page 1.
The third finds page 1 full, so A takes a new page:

```plain
page 1: [a4 a5 a6 a7]
page 3: [a8 __ __ __]

A: page_ids = [0, 1, 3]   page_lens = [4, 4, 1]   offset = 9
```

None of these appends touched page 0, or any of A's earlier tokens.

**B finishes.** Its page goes back on the free list. The data in page 2 stays
where it is, but nothing points to it anymore:

```plain
free list: [2]
```

**Request C arrives with a 5-token prompt.** The pool reuses page 2 before
creating anything new, then adds page 4 for the fifth token:

```plain
page 2: [c0 c1 c2 c3]
page 4: [c4 __ __ __]

C: page_ids = [2, 4]   page_lens = [4, 1]   offset = 5
```

A's pages are 0, 1, and 3, which are not next to each other in memory. That
is fine, because the page table records the order. To find token `t` of a
request:

```plain
page = page_ids[t // page_size]
slot = t % page_size

A, token 7: page_ids[7 // 4] = page_ids[1] = 1, slot 7 % 4 = 3
A, token 8: page_ids[8 // 4] = page_ids[2] = 3, slot 8 % 4 = 0
```

That lookup is the whole idea. Everything else in this chapter is
bookkeeping to keep the tables and the pool consistent.

## The Two Objects

### `TinyKvPagedPool`: One per Layer

The pool owns the actual memory for one transformer layer: one tensor for
keys and one for values, each shaped

```plain
[capacity, H, page_size, D]
```

so `key_pages[page_id]` is one page of keys for all `H` K/V heads. The pool
also keeps the free list of page ids.

Every request's cache for layer 0 shares layer 0's pool. Layer 1 has its own
pool, so the page ids in the two layers are unrelated: page 3 in layer 0 and
page 3 in layer 1 are different memory. Keeping layers in separate pools also
means a write in one layer never touches another layer's storage. Layer-local
ids also match what the attention kernel receives in Day 4: the K/V pages for
one layer.

The model creates the pools, one per layer, and keeps them in a list. Name it
`page_pools`: the batch benchmark looks for that attribute to read the pool
counters and to reset the pools after warmup.

### `TinyKvPagedCache`: One per Request per Layer

A request has one cache object per layer. Each holds its pool and its own
metadata:

- `page_ids`: its pages, in logical order;
- `page_lens`: how many valid tokens each page holds;
- `offset`: the total number of cached tokens;
- `page_size`: copied from the pool.

Two requests in the same layer share a pool but never share these lists. A
cache must not create its own pool; the model passes one in. A private pool
would cut the request off from pages other requests have freed.

## Growing the Pool Without Copying Everything

The pool's tensors have a fixed `capacity`. When the pool runs out of
capacity, it has to allocate a bigger tensor and copy the old pages into it.

If capacity grows by exactly one page each time, the copies add up fast. Going
from 1 to `p` pages copies `1 + 2 + ... + (p - 1)` pages, which is roughly
`p² / 2`. Instead, start with four pages and double the capacity whenever the
pool is full:

```plain
pages in use:   1-4    5    6-8    9    10-16   17 ...
capacity:        4     8     8    16      16    32
pages copied:    -     4     -     8       -    16
```

Each growth copies only the pages that already exist, and growths become
rarer as the pool gets bigger. In total, the copies add up to less than the
final capacity, and most allocations copy nothing.

Keep two numbers apart:

- `num_pages`: page ids handed out so far, including ones now on the free
  list;
- `capacity`: how many pages the tensors can hold.

After the fifth allocation, `num_pages` is 5 and `capacity` is 8. Whatever you
expose to attention as `key_pages` and `value_pages` should contain only the
`num_pages` pages that have been handed out, not the unused capacity.

## Writing Only the New Slice

When a token is appended, the pool should write `H × 1 × D` numbers into one
page. The obvious way to do that in MLX is a slice assignment:

```python
pages[page_id, :, start:end, :] = new_values
```

MLX arrays behave as values, though, and evaluation is lazy. That assignment
may be carried out by producing a whole new page tensor, which for a big pool
means copying every page to change a few numbers.

You will fix this with a small custom primitive, `paged_cache_update`. Its
output shares the input's memory instead of copying it, and its Metal kernel
launches one thread per element of the new slice: `H × new_tokens × D`
threads, no matter how big the pool is. Normally MLX assumes an array's
contents never change after it is computed. This primitive breaks that
assumption on purpose. It is safe here because the pool is the only owner of
its page tensors, and it replaces its reference with the returned array right
away, so later reads depend on the updated array. The only full copies left
are the doubling steps from the previous section.

## Appending a Chunk

Here is the append for one request in one layer, with a chunk of `S` new
tokens:

1. If the last page has free slots, write as many tokens as fit into it, and
   increase its `page_lens` entry.
2. While tokens remain, allocate a page, write up to `page_size` tokens into
   it starting at slot 0, and append its id and length.
3. Add `S` to `offset`.

A 6-token prompt with page size 4 becomes one full page and one page with two
tokens. The next decode token lands in the second page's slot 2, with no new
allocation. Allocate a page only when the last one is full. Allocating early
leaves empty slots in the middle of a request, and the pool has to grow
sooner than it needs to.

### Make Appends All-or-Nothing

An append can fail partway. A 7-token chunk with page size 4 needs two or
three writes, and the second write might fail. If the first write has already
updated `page_lens` and claimed a page, the cache now describes tokens it does
not hold, and the next attention step reads garbage.

Make every append succeed completely or change nothing:

- **Check first.** Before touching anything, reject a key and value that
  differ in shape, are not 4D `[1, H, S, D]`, use a dtype other than float32 or
  bfloat16, or do not match the pool's existing page shape and dtype. Raise
  `ValueError`.
- **Save, then restore.** Before writing, save the cache's lists and offset and
  the pool's state: its tensors, free list, and counters. If anything raises
  during the append, put all of it back and re-raise.

Restoring the pool's old tensors works even though `paged_cache_update` wrote
into shared memory. The slots written before the failure lie past the
restored `page_lens`, so nothing reads them, just like the leftover data in a
freed page.

## Releasing, Rewinding, and Resetting

**`release()`** returns every page the cache holds to the pool's free list and
clears its metadata. Freeing a page does not erase it. `page_ids` and
`page_lens` decide which slots are live, and the next owner overwrites the
slots it uses.

**`rewind(n)`** removes the newest `n` tokens, keeping the rest in place:

```plain
A before: page_ids = [0, 1, 3]   page_lens = [4, 4, 1]   offset = 9
rewind(3)
A after:  page_ids = [0, 1]      page_lens = [4, 2]      offset = 6
          page 3 goes back to the free list
```

Free each page that no longer holds any token, then shorten the last
remaining page's length. Treat the edges carefully: `rewind(0)` changes
nothing, rewinding the whole length is the same as `release()`, and a value
that is negative, larger than `offset`, or not an `int` (including `True`)
must raise `ValueError` without changing anything. Week 2 Day 1 gave the dense
cache the same rule, and the tests check both caches. The optional
[speculative decoding chapter](./week3-optional-speculative-decoding.md) uses
rewind to discard rejected draft tokens.

**`TinyKvPagedPool.reset()`** drops the pool's storage completely: capacity,
page count, and free list all go back to zero. Refuse to reset while any page
is still in use. The benchmark calls `reset()` after its warmup runs, so the
measured run starts from an empty pool.

## Batching Requests With Pages

`BatchingKvCache` holds one layer cache per decode slot. Add
`update_and_fetch_paged`. It takes a batch of new keys and values, appends
each active row to its request's cache, and returns a `PagedKvMetadata` that
describes the whole batch:

- `key_pages`, `value_pages`: the layer pool's page tensors;
- `block_table`: one row of page ids per slot, padded with `-1` to the
  longest row;
- `context_lens`: each slot's `offset` after the append;
- `page_size` and `mask`.

For requests A and C from the example:

```plain
block_table  = [[0, 1,  3],
                [2, 4, -1]]
context_lens =  [9, 5]
```

This is the input Day 4's attention kernel reads. For now, the tests use it to
check that each row maps back to the right tokens.

The batch append must be all-or-nothing as well, across all rows. Before
appending anything, check that every active cache is a `TinyKvPagedCache`,
that they all share one pool, and that each row's append is valid. Then save
the pool and every active cache, append row by row, and restore everything if
any row fails. A batch whose rows use different pools should fail before row 0
changes.

`TinyKvPagedCache` itself gets matching single-request methods:
`block_table(max_pages=None)`, `context_lens()`, `paged_metadata(...)`, and
`update_and_fetch_paged(...)`, which appends and then returns the metadata.

## Checking Against the Dense Cache

Your Week 2 `TinyKvFullCache` already produces correct K/V, so use it as the
answer key. Add `gather_dense()`, which walks `page_ids`, takes the first
`page_lens[i]` slots of each page, and concatenates them into a
`[1, H, offset, D]` tensor. Then `update_and_fetch` (the old dense interface)
can append and return the gathered tensors. Any caller that worked with the
dense cache keeps working, and you can compare the two caches after every
append.

This copies a request's history before every attention call, which is the
same cost you started the chapter with. That is deliberate: this checkpoint
only changes how K/V is stored, so any bug you find is a storage bug. Day 4
removes the gather by passing `block_table` and `context_lens` to attention
instead.

## Task 1: Pages, Pool, and Request Cache

```
src/tiny_llm/paged_kv_cache.py
src/tiny_llm/kv_cache.py
```

Implement `TinyKvPagedPool` and `TinyKvPagedCache` in `paged_kv_cache.py`, and
`BatchingKvCache.update_and_fetch_paged` in `kv_cache.py`.

The starter lists most of the methods. The tests also use a few attributes it
does not declare, so add them as well:

| Object | What the tests use |
|---|---|
| `TinyKvPagedPool` | `page_size`, `num_pages`, `num_free_pages`, `capacity`, `storage_nbytes`, `allocate_page()`, `read_page(page_id)`, `write_page_slice(page_id, start, key, value)`, `free_page(page_id)`, `reset()` |
| `TinyKvPagedCache` | `pool`, `page_ids`, `page_lens`, `offset`, `num_pages`, `key_values` (the gathered dense pair, or `None` when empty), `gather_dense()`, `update_and_fetch(...)`, `update_and_fetch_paged(...)`, `block_table(...)`, `context_lens()`, `paged_metadata(...)`, `materialize()`, `rewind(n)`, `release()` |

`materialize()` evaluates the pool's page tensors. Day 2's scheduler already
calls it after each prefill chunk, which keeps MLX's lazy graph from growing
without bound.

For this task, `write_page_slice` may use plain MLX slice assignment. It is
correct, just not cheap. Task 2 replaces it.

The benchmark also prints a few optional pool counters if your pool defines
them: `reused_page_allocations`, `storage_growths`,
`copied_pages_on_growth`, and `copied_bytes_on_growth`. They are a good way
to see the doubling schedule at work.

```bash
pdm run test --week 3 --day 3 -- -k task_1
```

These tests compare your cache with `TinyKvFullCache` across appends that
cross page boundaries; check page reuse after release, doubling growth, and
`reset()`; and check that rejected or failed appends, including batch appends
and mixed pools, leave every cache and the pool unchanged. They also cover
rewind against the dense cache, `block_table`, `materialize`, and independent
metadata for two requests that share a pool.

## Task 2: Write Only the New Slice

```
src/extensions/src/paged_attention.cpp
src/extensions/src/paged_attention.metal
src/tiny_llm/paged_kv_cache.py
```

The C++ and Metal files, the header declaration, the Python binding, and the
CMake entries for `paged_cache_update` already exist. Replace the stubs:

- `tiny_llm_ext::paged_cache_update(pages, values, page_id, start)` checks its
  inputs and returns a lazy array with the same shape and dtype as `pages`.
  Require `pages` to be `[P, H, page_size, D]` and `values` to be
  `[1, H, length, D]`, with matching `H`, `D`, and dtype (float32 or
  bfloat16). Require `0 <= page_id < P` and
  `0 <= start` with `start + length <= page_size`. Throw on anything else.
- `PagedCacheUpdate::eval_cpu` throws. The operator is GPU-only.
- `PagedCacheUpdate::eval_gpu` checks that both inputs are row-contiguous,
  makes the output share the `pages` buffer, and launches
  `paged_cache_update_kernel` over the `H × length × D` new elements.
- `paged_cache_update_kernel` in `paged_attention.metal` copies each element
  of `values` to its slot in page `page_id`, starting at `start`.

Do not add a second primitive or binding. Rebuild, then switch
`TinyKvPagedPool.write_page_slice` to call the extension for keys and for
values, passing contiguous chunks:

```bash
pdm run build-ext
pdm run test --week 3 --day 3 -- -k task_2
```

The Task 2 tests check that only the requested slice changes, for float32 and
bfloat16; that bad shapes, dtypes, page ids, and slot ranges are rejected; and
that removing a request from `BatchingKvCache` returns its pages, so the next
request reuses them without growing the pool.

## Task 3: Put the Pages in the Model

```
src/tiny_llm/qwen3_week3.py
```

Update `Qwen3ModelWeek3.__init__`, `Qwen3ModelWeek3.create_kv_cache`, and
`Qwen3MultiHeadAttention.__call__`:

- In `__init__`, create one `TinyKvPagedPool(page_size=page_size)` per layer
  and keep them in `self.page_pools`.
- `create_kv_cache()` returns one new `TinyKvPagedCache` per layer, each
  attached to that layer's pool.
- When the model is built with `enable_paged_attention=False`, attention calls
  the cache's dense `update_and_fetch` and runs the same dense attention as
  before on the gathered K/V. Leave `paged_attention` alone; it belongs to
  Day 4.

Keep the projection setup from Day 1: `use_mlx_quantized_linear=True` stays
the Week 3 default, so the projections run through `mx.quantized_matmul`,
while embedding, normalization, activation, RoPE, cache, and attention stay
your own code.

```bash
pdm run test --week 3 --day 3 -- -k task_3
```

The Task 3 tests compare the Week 3 model with your Week 2 model on a full
prompt and token by token during decode, with tied and untied output heads.
They also build two requests from one model, batch them, and check the
returned `block_table` and `context_lens`.

Then run the whole day's tests and the full model:

```bash
pdm run test --week 3 --day 3

pdm run main --solution tiny_llm --loader week3 \
  --disable-paged-attention --model qwen3-0.6b

pdm run bench --solution tiny_llm --loader week3 \
  --disable-paged-attention --batch-decode --model qwen3-0.6b
```

## What to Look For in the Benchmark

The batch benchmark reads your caches' `page_ids`, `page_lens`, `page_size`,
and `pool`, and your model's `page_pools`, so it can report on paging without
any extra code from you. Run the same command with `--solution tiny_llm_ref`
to see the reference numbers next to yours.

**Reuse.** `Reused page allocations` counts pages handed to a new request
after an earlier one released them. `Peak live KV pages` and
`Peak KV capacity pages` show how many pages were in use, and how many the
pool had room for, at their peaks. If live pages keep climbing while requests
finish, something is not releasing its pages.

**Growth.** `Page-pool growths` and `Pages copied during pool growth` should
show a handful of doubling steps, not one growth per page. These come from the
optional pool counters in Task 1, so they read 0 if you did not add them.

**Wasted slots.** Only a request's last page can be partly empty, so each
request-layer cache wastes at most `page_size - 1` slots. The benchmark finds
the moment when the sum of those unused last-page slots was largest, and
reports it as a fraction of all slots in live pages at that same moment
(`Tail-waste snapshot fraction`), along with its size in bytes. Spare pool
capacity is not counted in that fraction; it shows up in the capacity
counter instead.

**Copies that remain.** Attention still receives gathered dense K/V in this
checkpoint, so the batch staging copy from the start of the chapter is still
there. Day 4 removes it.

Paging does not make a single request faster. A page table adds a lookup,
and this checkpoint still copies K/V for attention. What it gives a server is
memory that is handed out as requests grow, returned when they finish, and
reused by the next request. Apple silicon's unified memory removes the copy
between CPU and GPU memory, but not allocation, fragmentation, or copying
inside GPU-visible memory, so these savings still apply.

One run with a fixed batch shows how much K/V memory you saved. It does not
show that the server can accept more requests at once. To claim that, you
would need to cap memory and measure how many requests fit.

Next, in Day 4, attention reads `block_table` and `context_lens` directly and
the dense gather goes away.

{{#include copyright.md}}
