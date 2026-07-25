# 🚧 Week 3: Build a Mini vLLM

> 🚧 This overview and chapters carrying the same marker are under review and
> may change.

Week 3 turns the optimized single-request model into a multi-request serving
engine. Students add scheduling, request-owned cache state, shared page pools,
and the runtime metadata needed to read noncontiguous K/V directly. The final
model uses one page-aware attention interface with separate schedules for
one-token decode and multi-token prefill.

As in Week 2, **MLX** names the framework or its production operators, the
**reference solution** names `tiny_llm_ref`, and **your solution** names the
code you write in `tiny_llm`.

## What We’ll Cover

- Continuous batching and request-slot reuse
- Chunked prefill and scheduler fairness
- Paged KV storage and page-walking attention
- Paged FlashAttention for long prefill
- Optional speculative decoding over rewindable caches
- Optional Mixture-of-Experts model support

The ordering is intentional. Day 1 batches independent request states. Day 2
splits long prefills so they cannot monopolize the scheduler. Day 3 replaces a
growing dense cache with fixed-size pages while retaining a dense-gather
compatibility path. Day 4 removes that gather by teaching attention to walk the
page table directly with a correctness-first schedule. Day 5 then tiles that
same page-walking operation with Week 2's matrix fragments. Page translation
is therefore introduced before it is optimized.

These five days form the required path in your solution. The final model in
your solution runs paged FlashAttention for long prefill and the paged vector
kernel for short queries.
Both schedules read the same page pool through the same block-table interface;
neither rebuilds dense K/V.

Paged attention is not an automatic single-request latency win. It primarily
improves serving capacity, cache reuse, and batching; page-table indirection
can make one request slower. This week measures those tradeoffs rather than
assuming an algorithm with a production name is automatically fast. Each
chapter ends with a focused measurement, while the
[performance appendix](./appendix-performance.md) records the matched
chapter-by-chapter results.

Speculative decoding follows the paged-attention chapters because rejection
needs a precise cache rewind operation, and multi-token verification needs the
page-aware long-query path. MoE is independent of the cache and scheduler, so
it remains an optional model extension and is not required to complete Week 3.

{{#include copyright.md}}
