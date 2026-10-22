# 🚧 Week 3 Day 3: Paged KV Cache

> 🚧 This chapter is under review and may change.

Your Day 2 engine can admit a request, prefill its prompt in chunks, decode
beside other requests, and reuse its slot when it finishes. Today, keep that
serving loop and change what a request owns: a list of small cache handles
pointing into shared page pools, one pool for each transformer layer.

Start from **your completed Week 2 model and Week 3 Days 1–2 solution**. The
Week 3 starter still contains methods for you to implement; selecting
`--loader week3` does not fill them in. Before changing storage, check the
chunked-prefill behavior you are carrying forward:

```bash
pdm run test --week 3 --day 2
pdm run batch-main --solution tiny_llm --loader week2 --model qwen3-0.6b
```

The second command uses Day 1's dense batching model. Watch requests enter,
produce text, finish, and make room for later requests. It is a serving smoke
check, not a controlled throughput comparison. Keep this working path while
you build its paged-cache replacement.

Use the course's configured PDM environment and Apple-silicon GPU setup. The
native cache-write primitive below requires rebuilding your learner extension.
The storage and small-model tests use generated fixtures; the generation and
serving commands at the end additionally need the named model assets.

## The Behavior You Will Build

Suppose request A has six cached tokens and pages hold four tokens each:

```plain
request A, layer 0 -> page ids [7, 2], valid lengths [4, 2]
request B, layer 0 -> its own page ids and valid lengths
                     both handles use layer 0's pool
```

A's next token fills the third slot of page 2. When A finishes, its pages return
to the pool. B or a newly admitted request can reuse them. B's logical sequence
does not change when A appends or releases storage.

You will first **gather these pages into dense K/V and use dense attention**.
That gives the new allocator, mapping, and request lifecycle a familiar
correctness check. Day 4 will make attention read the pages directly; Day 5
will tile that computation. You do not need either future attention kernel to
complete today's storage path.

There are four cumulative checkpoints:

| Checkpoint | Learner-owned work | First useful feedback |
| --- | --- | --- |
| 1. Physical writes and pool | Slice-sized native update, page allocation, geometric storage growth | The `task_2_paged_cache_update` test families |
| 2. One request's logical cache | Tail append, valid lengths, dense gather | The focused dense-reconstruction test |
| 3. Serving lifecycle and metadata | Release, rewind, materialize, storage descriptors, batched validation and rollback | All storage tests: `task_1 or task_2` |
| 4. Dense Week 3 model | Dense model/block/MLP assembly, per-layer pools, dense-gather attention | `task_3`, then the complete Day 3 suite and serving entry points |

The supplied test names predate this implementation order: `task_2` includes
both the native write and batch removal. Use the filters below rather than
expecting the complete suite to pass after the first step. `pdm run test`
copies the supplied day test into `tests/` before running it against
`tiny_llm`; make changes in your implementation, not in that copied test.

**📚 Readings**

- [vLLM Paged Attention Design](https://docs.vllm.ai/en/v0.18.0/design/paged_attention/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

## 1. Give the Pool a Slice-Sized Write

### Why change the dense layout?

A dense request cache stores one growing K/V sequence per layer. For batching,
Day 1's adapter repacks those sequences into rectangular tensors:

```plain
keys:   [B, H_kv, S_max, D]
values: [B, H_kv, S_max, D]
mask:   [B, 1, L, S_max]
```

Here `B` is the number of batch rows, `H_kv` the number of K/V heads, `L` the
new chunk length, and `D` the head dimension. A request with 17 cached tokens
adds only one new token during decode, yet a dense repack presents all 18
logical positions again. Across many live requests, repeatedly moving history
can become substantial work.

Paging separates a sequence's logical order from its physical placement.
Represent each layer's backing K/V as two slabs:

```plain
key storage:   [capacity, H_kv, page_size, D]
value storage: [capacity, H_kv, page_size, D]
```

A page is a fixed number of token slots for every K/V head. Page ids address
these slabs; they are not token positions. Start with the model's default
`page_size=128`, and use small pages such as four tokens when reasoning about
boundaries and writing tests.

### Implement the existing native primitive

Before building append logic, make a write touch only the new physical slice.
MLX is lazy, and a Python assignment such as
`pages[page_id, :, start:end, :] = values` does not by itself establish that
the device will update only that slice. It can produce a new full-sized array.

The learner extension already declares and registers `paged_cache_update`.
Replace its Day 3 stubs in:

```plain
src/extensions/src/paged_attention.cpp
src/extensions/src/paged_attention.metal
```

Complete `tiny_llm_ext::paged_cache_update`, `PagedCacheUpdate::eval_cpu`,
`PagedCacheUpdate::eval_gpu`, and `paged_cache_update_kernel`. The header,
Python binding, CMake entry, and source files already exist. Keep this API;
you do not need a second primitive or the Day 4 `paged_attention` implementation.
This primitive is GPU-only, so its CPU evaluator should report that limitation.

The operation accepts pages shaped `[capacity, H_kv, P, D]` and a contiguous
chunk shaped `[1, H_kv, N, D]`, where `P` is `page_size`. Validate the rank,
matching head dimensions and dtype, page id, and slice bounds before encoding
a write. Support the supplied float32 and BF16 cases. One call writes within
one physical page, so `0 <= start` and `start + N <= P` must hold.

For head `h`, new-token index `t`, and dimension `d`, the destination element
in row-major storage is:

$$
\operatorname{destination}(h,t,d)
= ((\operatorname{page\_id}\,H_{kv}+h)P+\operatorname{start}+t)D+d.
$$

Launch work for `H_kv * N * D` elements, with a bounds check for the final
threadgroup. Each thread copies the corresponding chunk value into this
location. There is no reason to visit other pages or the retained prefix.

The output must alias the page storage rather than allocate and copy a new
slab. Keep both page storage and incoming values as primitive graph inputs,
register the device buffers with the command encoder, and retain the returned
array as the pool's updated storage. Attention and later writes must depend on
that returned array. This is an explicit mutation boundary for cache state,
not an immutable snapshot of every historical version of the array.

This design relies on page ownership: two live request caches in the same
layer must never own the same page id. It also relies on the existing stream
and materialization boundaries before a finished request's storage is reused.
Aliasing alone does not make arbitrary overlapping writes safe.

Rebuild, then run only the primitive's tests:

```bash
pdm run build-ext
pdm run test --week 3 --day 3 -- -k task_2_paged_cache_update
```

These checks cover the requested slice, untouched surrounding storage, supported
dtypes, and invalid inputs. They do not yet require a complete page allocator
or Week 3 model. If a slice test fails, inspect the physical index and aliasing
contract before adding request-level machinery.

### Implement `TinyKvPagedPool`

Now fill in the pool in `src/tiny_llm/paged_kv_cache.py`. It needs an allocator,
a free list, and K/V storage. The model will own one pool per transformer layer:

```plain
model
  layer 0 pool <- request A cache[0], request B cache[0], ...
  layer 1 pool <- request A cache[1], request B cache[1], ...
  ...
```

Layer 0 and layer 1 can both allocate page id 0: those ids refer to different
buffers. A write or growth in one layer should not replace a slab containing
all the other layers' storage. Within a layer, request handles share the pool
but own disjoint live pages.

Complete the starter's constructor, `allocate_page`, `free_page`, `read_page`,
`write_page_slice`, `num_pages`, and `num_free_pages`. Add the storage properties
and helpers needed by this interface and its tests; the starter does not supply
a finished allocator. In particular, expose `capacity`, `key_pages`,
`value_pages`, and `storage_nbytes`, and provide `reset()` for an empty pool.

Keep these counts distinct:

| Value | Meaning |
| --- | --- |
| Pool `capacity` | Number of page slots in the physical backing slabs |
| Pool `num_pages` | Number of page ids created so far, including ids now on the free list |
| Pool `num_free_pages` | Created ids available for another allocation |
| Request cache `num_pages` | Number of pages currently owned by that one request-and-layer handle |

Thus a pool's live page count is `num_pages - num_free_pages`, not necessarily
`num_pages`. Freeing a page preserves its id and moves it to the free list;
allocating should reuse a free id before creating another. Reject a direct
double free. Old bytes can remain in a freed page because a new owner's valid
lengths determine what may be read.

Initialize storage using the first valid chunk's head count, head dimension,
and dtype. Subsequent chunks must agree. `write_page_slice` checks that its page
is allocated and that the write fits, then uses `paged_cache_update` separately
for K and V and retains both returned arrays.

### Grow capacity geometrically

Do not make allocating every new id copy all old storage. Growing an exact-size
slab from `p` to `p + 1` pages copies approximately

$$
1 + 2 + \cdots + p = \frac{p(p+1)}{2}
$$

old pages over the sequence of growth events. That is quadratic in the final
page count. Instead, start with capacity four and double when needed. The
copies at growth events form a geometric series, so total copied storage grows
linearly with the final capacity. Between growth events, only appended slices
need writing. Preserve existing K/V and stable page ids when replacing a slab.

For example, creating the fifth id may grow capacity from four to eight.
`key_pages` and `value_pages` should expose the first five created ids, while
the backing allocation has eight page slots. Free ids among those five remain
addressable for later reuse; they are not live sequence contents.

Track reused allocations and storage-growth/copy counters separately. They
will help explain the serving run at the end. `reset()` is different from
freeing a request: when no ids are live, it removes retained pool storage and
resets allocator state and counters. Reject a reset with live pages. The
benchmark uses this operation to keep warmup allocation out of the measured
storage counters.

## 2. Map One Request's Tokens onto Pages

Implement `TinyKvPagedCache` in `src/tiny_llm/paged_kv_cache.py`. Its constructor
receives a model-owned pool. It must not create a private pool for each request,
or finished requests would be unable to donate capacity to later requests.

A handle owns `page_ids`, `page_lens`, and `offset`. For `P=4`, consider:

```plain
page_ids  = [12, 5, 3]
page_lens = [ 4, 4, 2]
offset    = 10

page 12 -> logical tokens 0..3
page  5 -> logical tokens 4..7
page  3 -> logical tokens 8..9
```

The sequence remains length ten. For logical token position `t`, its page-table
column and within-page slot are:

$$
j = \left\lfloor \frac{t}{P} \right\rfloor,
\qquad u = t \bmod P,
\qquad \operatorname{physical\_page}(t) = \operatorname{page\_ids}[j].
$$

A normal append-only handle has full pages except possibly its tail. Maintain
`offset == sum(page_lens)` and `num_pages == len(page_ids)`; for a nonempty
sequence, the number of pages is `ceil(offset / P)`. Capacity in the unused
suffix of a page is not part of the logical sequence.

### Fill the tail before allocating

Suppose a prefill chunk has six tokens:

```plain
chunk = [t0 t1 t2 t3 t4 t5]
page 7 <- [t0 t1 t2 t3]  # 4 valid tokens
page 2 <- [t4 t5 _  _ ]  # 2 valid tokens
```

The handle becomes `page_ids=[7, 2]`, `page_lens=[4, 2]`, `offset=6`.
A later chunk first fills page 2's unused suffix, then allocates pages for any
remainder. For each write, choose the smaller of the remaining input and the
available slots in the current page. Send only that chunk slice to the pool.

Decode is the same operation with a one-token chunk. With `P=4` and
`offset=9`, the handle `[12, 5, 3]` has valid lengths `[4, 4, 1]`. Appending token
at logical position 9 updates page 3's second slot. Pages 12 and 5 retain their
contents and ids. Keep chunked-prefill and decode on this one append path so
boundary cases have one implementation.

### Gather only valid prefixes

Complete `gather_dense` and `update_and_fetch`. Read pages in **page-table
order**, take only each page's valid prefix, and join along the token dimension.
Physical id order is irrelevant: sorting `[12, 5, 3]` would scramble the sequence.
Never expose the unused suffix of the final page or stale data from an old
owner.

The gathered K/V has shape `[1, H_kv, offset, D]`. Preserve the cache interface's
returned offset and mask, including the existing rectangular causal-mask
behavior for chunks and batching. Also expose a `key_values` view for callers
and tests: it is `None` for an empty cache and the gathered pair otherwise.
The pages remain the source of truth; a gathered tensor is a view of logical
contents for a consumer, not a second independently updated cache.

Before extending this to the scheduler, check dense reconstruction:

```bash
pdm run test --week 3 --day 3 -- -k test_task_1_paged_cache_matches_full_cache
```

This test appends several chunks across page boundaries and compares the
logical sequence with `TinyKvFullCache`. It needs the constructor, append, pool
read/write, and dense gather. The independent-handle test also calls the paged
metadata interface, so run it with the complete storage checkpoint below,
after implementing that interface.

## 3. Complete the Serving Storage Contract

A cache must remain usable when requests finish, a chunk is rejected, or a
caller drops recently appended tokens. These operations belong to today's
storage abstraction, even though later chapters will use them in new ways.

### Release, materialize, and rewind

Implement `release()` so it returns every owned page id to the pool and clears
the handle's page ids, valid lengths, and offset. Releasing an already empty
handle should be harmless. It does not reset the shared pool: other requests
may still be using it, and later requests should reuse its free pages.

Keep the scheduler representation-independent. A `Request` owns one cache
handle per layer; it need not know how pages are stored. In
`src/tiny_llm/kv_cache.py`, extend `BatchingKvCache.remove_request` to release
the removed handle. Preserve cleanup for completion and aborted pending
prefill in `src/tiny_llm/batch.py`, releasing every layer's handle. A reusable
batch slot is not permission to reuse pages still needed by queued GPU work.

Implement `materialize()` by evaluating the live pool writes at the existing
chunk/scheduler boundaries without gathering the whole sequence. Materializing
must preserve the logical contents. This prevents an ever-growing lazy append
graph while keeping the purpose of the boundary clear: evaluation of current
cache state, not construction of dense attention inputs.

Add `rewind(n)` to both `TinyKvPagedCache` and the dense `TinyKvFullCache`
interface in `src/tiny_llm/kv_cache.py`. The Day 3 tests cover both. Rewind removes
the newest `n` logical tokens: zero changes nothing; removing the whole logical
length empties the cache; a partial rewind retains exactly the old prefix.
Reject booleans, non-integers, negative values, and lengths beyond the current
offset before changing state.

For pages, free whole pages beyond the retained prefix and shorten the final
remaining page's valid length. The next append must overwrite the newly invalid
suffix rather than resurrect old tokens. For a dense cache, retain the same
logical-prefix behavior whether storage is bounded or concatenated. The
unbounded dense cache used by these rewind tests clears `key_values` to `None`
when emptied; a bounded dense cache may retain its allocation while setting
its logical offset to zero. The optional speculative-decoding chapter will
use rewind, but its algorithm is not required here.

### Build storage metadata now; consume it in Day 4

The Day 3 suite also checks `block_table`, `context_lens`, `paged_metadata`, and
`update_and_fetch_paged`. Implement these **storage descriptors now**, alongside
the dense-gather path. Constructing a descriptor does not execute paged
attention. Day 4 will use these values as attention-kernel inputs; its first
task is also an opportunity to check and extend the descriptor wiring you
establish here.

For one cache, `block_table(max_pages)` is an int32 row of page ids shaped
`[1, max_pages]`, padded with `-1` beyond the owned pages. Its unpadded width is
the request's page count. `context_lens()` is an int32 array shaped `[1]`
containing `offset`. `PagedKvMetadata` bundles the shared K/V page arrays,
block table, context lengths, page size, and mask.

When appending within the tail, the block-table entries do not change, but the
context length does. When a new page is allocated, the table gains an entry.
Releasing or rewinding must likewise update the descriptor's logical view.
Keep these two kinds of change separate when deciding which metadata can be
reused.

Extend `BatchingKvCache.update_and_fetch_paged` in
`src/tiny_llm/kv_cache.py` to describe multiple live handles:

```plain
key_pages, value_pages: shared layer-pool arrays
block_table:            [B, max_pages] int32
context_lens:           [B] int32
mask:                  the batch's existing attention mask
```

Each active row refers to that request's page order and context length.
Inactive rows have zero context length and `-1` table entries. All active
handles must use the **same layer pool**; page id 3 in another pool is not the
same storage. Keep the established masks and per-row offsets from Days 1–2.
This adapter prepares storage metadata; it must not invoke the future kernel.

### Validate before changing any request

A rejected chunk must leave the old logical cache usable. Validate matching
K/V shapes, the single-request batch dimension, positive head/token/dimension
sizes, supported matching dtype, and compatibility with existing pool storage
before allocation or writes. At batch level, validate all active rows, shared
pool identity, and length limits before mutating row zero.

The supplied tests also inject failures after part of an append has been
prepared. Design rollback for pool allocation/free-list state and each affected
handle's page ids, lengths, offset, and storage references. A multi-page append
must not leak a page or expose a partially accepted chunk; a batched failure
must restore every row, not just the row that raised.

Append writes go to new pages or the unused suffix of an existing tail. The
previous valid prefix must remain intact, which makes restoring its logical
metadata possible without copying the entire history for every append. Keep
that invariant explicit when handling an error; this is not a general
transaction system for arbitrary overlapping device mutations.

Run the complete storage checkpoint:

```bash
pdm run test --week 3 --day 3 -- -k 'task_1 or task_2'
```

At this point, expect coverage of dense reconstruction, geometric growth and
reuse, reset, slice writes, invalid-input non-mutation, multi-page and batched
rollback, materialization, block tables, release on slot removal, and dense and
paged rewind. If these fail, stay with the storage contract before debugging
model logits.

## 4. Connect the Dense Week 3 Model

The remaining learner work is in `src/tiny_llm/qwen3_week3.py`. Its attention,
MLP, transformer block, and model methods are stubs; completing only the model
constructor and cache factory does not produce a runnable model. Carry forward
your working dense Qwen3 computation and make the cache substitution at its
existing attention boundary.

Work through the complete dense chain:

1. Complete `Qwen3MLP.__init__` and `__call__` with the existing gate/up,
   activation, and down-projection behavior.
2. Complete `Qwen3MultiHeadAttention.__init__` and `__call__`: projections,
   Q/K normalization, multi-offset RoPE, cache update, dense attention, and
   output projection.
3. Complete `Qwen3TransformerBlock.__init__` and `__call__`, retaining both
   normalization/residual stages and forwarding the correct layer cache,
   offsets, and mask.
4. Complete `Qwen3ModelWeek3.__init__`, `create_kv_cache`, and `__call__`:
   embeddings, dense layers, final normalization, tied or untied output head,
   and the existing `logits_to_keep` behavior. Construct one page pool per
   layer and return a fresh handle for each pool on each cache-factory call.

Retain Day 1's dense projection boundary. Quantized linear projections use the
`mx.quantized_matmul` seam, with `use_mlx_quantized_linear=True` as the dense
Week 3 default. Embedding lookup, normalization, activation, RoPE, cache,
attention, and scheduling remain course-owned. Keep the opt-out as a
correctness/benchmark ablation rather than silently changing the model's
projection path. This chapter uses dense Qwen3; the optional MoE chapter owns
its separate router/expert implementation and projection contract.

### Keep the attention equation unchanged

Construct the model with `enable_paged_attention=False` for today's path. For
each layer, write new K/V through the cache, gather the logical sequence, and
apply the same dense attention computation:

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\!\left(\frac{QK^\mathsf{T}}{\sqrt{D}}+M\right)V.
$$

Keep grouped-query head handling and the causal/padding mask `M` from the
dense path. Changing where K/V lives does not change which logical positions
a query may attend to. For batching, `BatchingKvCache.update_and_fetch` still
builds dense rows from the live handles. This means **Day 3 still gathers and
repacks historical K/V**; allocation and reuse are paged, while attention input
construction remains a visible cost for Day 4 to remove.

Two calls to `create_kv_cache()` must produce distinct request metadata at
every layer, backed by the same corresponding pools. An append in layer 0
must not advance layer 1, and request A must not change request B's offset.
The small model tests check this ownership boundary as well as full-prompt
and incremental-decode agreement with the Week 2 dense model, using the
supplied numerical tolerances.

Check the model only after storage is working:

```bash
pdm run test --week 3 --day 3 -- -k task_3
pdm run test --week 3 --day 3
```

The first command narrows failures to model assembly and cache integration.
The second is the cumulative Day 3 gate, including the storage metadata that
Day 4 will consume. Passing only a generation smoke check is not a substitute
for these lifecycle and mapping tests.

## Use the Same Serving Loop with Paged Storage

Run the normal generation entry point through the dense-gather path:

```bash
pdm run main --solution tiny_llm --loader week3 \
  --disable-paged-attention --model qwen3-0.6b \
  --prompt 'Explain why a finished request can return KV pages to a shared pool.' \
  --max-tokens 64
```

Look for a coherent answer to the visible prompt, then inspect storage with a
small continuous-batching trace. The benchmark supports the dense-gather flag
and drives the shared-model request lifecycle from Days 1–2:

```bash
mkdir -p "$HOME/tiny-llm-week3-results"
RUN_DIR=$(mktemp -d "$HOME/tiny-llm-week3-results/paged-cache-XXXXXX")
pdm run bench --solution tiny_llm --loader week3 \
  --disable-paged-attention --batch-decode --model qwen3-0.6b \
  --num-seqs 8 --batch-size 4 --prefill-step 128 \
  --min-input-len 64 --max-input-len 256 \
  --min-output-len 32 --max-output-len 32 --seed 0 --warmup 1 \
  --json-output "$RUN_DIR/metrics.json"
```

Use `bench` for this Day 3 batching check: `batch-main --loader week3` does not
expose `--disable-paged-attention` and would select the later attention path.
Keep the flag above until you implement Day 4's kernel.

The benchmark excludes model loading and runs a warmup. After warmup it
synchronizes and resets the empty page pools before the measured batch, so
storage growth and reuse counters describe that measured trace rather than
retained warmup capacity. During the trace, finished requests release pages
for subsequent requests. The pool may retain backing capacity between these
requests; freeing a page does not promise that the process returns memory
to the operating system.

Read the output in terms of what this implementation actually does:

- Request completion and useful-token throughput describe the serving trace.
  They are not proof that storage alone caused a speed change.
- Reused page allocations show that request completion made storage available
  to later requests. Growth/copy counters show when the physical slabs had to
  expand despite slice-sized appends.
- Live pages and capacity pages describe different quantities: valid request
  ownership versus the physical allocation retained by the pool.
- Dense K/V gathering and batch repacking still cost work. Direct page reads
  are the next chapter's change, not an optimization already supplied here.

### Measure tail waste at one snapshot

For a nonempty request-and-layer cache with page size `P`, tail waste is
`P - page_lens[-1]` token slots. Full pages contribute zero waste. Sum over the
nonempty live caches at a single snapshot:

$$
\operatorname{tail\_waste\_fraction}
= \frac{\sum_{c\ \mathrm{live}} (P-\operatorname{last\_page\_fill}_c)}
       {\sum_{c\ \mathrm{live}} \operatorname{num\_pages}_c\,P}.
$$

The benchmark finds the snapshot with the largest aggregate unused tail-slot
count and uses the live-page denominator **from that same snapshot**. It also
reports unused-slot bytes. Do not divide a peak waste count by a live-page
count from another time. For one layer, a physical K/V page occupies
`2 * H_kv * P * D * bytes_per_element`; this converts slot waste to bytes for
that layer's storage dtype.

Unused physical pool capacity is outside this tail-waste fraction. Keep it
visible in the separate live-page, capacity-page, and storage-byte counters.
A pool can have low tail waste while retaining substantial free capacity.

Apple silicon's unified memory removes the discrete-device transfer boundary;
it does not remove allocation, fragmentation, or copying inside GPU-visible
memory. Paging gives the serving engine a way to grow requests without
reserving every request's maximum length and to reuse freed capacity. It does
not guarantee lower single-request latency. This fixed workload illustrates
KV-storage headroom and lifecycle behavior; showing that more requests can be
admitted requires a separate memory-capped admission sweep.

## Check Your Design Before Day 4

Trace these cases through your implementation:

1. A chunk partly fills a tail and then needs two new pages. Which lengths and
   allocator changes must be restored if its second page write fails?
2. Two requests in one layer both have a logical token at position nine.
   Which metadata keeps their physical destinations distinct?
3. A request finishes while another remains active. Why must releasing its
   handles leave the shared pool and the other request's page table intact?
4. A rewind leaves two valid slots in the tail. How does the next append avoid
   reading the old, now-invalid suffix?
5. A pool grows from four to eight page slots. Which page ids remain stable,
   which buffers change, and which counters should record copying?
6. Why can the Day 3 cache descriptors pass their tests while the Day 4
   attention kernel remains unimplemented? Where does dense gathering still
   happen in today's model?

You now have a storage contract that can be checked independently of indirect
attention reads: shared per-layer pools, request-owned mappings, valid prefixes,
release/reuse, and descriptors. In [Day 4](week3-04-paged-attention-part2.md), use
those same mappings to read K/V directly during attention instead of gathering
a dense history first.

{{#include copyright.md}}
