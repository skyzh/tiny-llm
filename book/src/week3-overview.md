# 🚧 Week 3: Build a Mini vLLM

> 🚧 This overview and chapters carrying the same marker are under review and
> may change. The optional MoE chapter is reviewed course material.

Week 3 takes the optimized single-request model from Week 2 and builds a serving
engine around it. The Week 2 operator interfaces remain intact; this week adds
multi-request cache ownership, scheduling, and runtime metadata. Week 2 has
already introduced BF16 model storage, online softmax, and SIMD-matrix
fragments. Week 3 adds page translation before combining those pieces into
paged FlashAttention.

## What We’ll Cover

- Continuous batching and request-slot reuse
- Chunked prefill and scheduler fairness
- Paged KV storage and page-walking attention
- Paged FlashAttention for long prefill
- Optional serving and scheduling experiments
- Optional MoE and speculative-decoding extensions

The ordering is intentional. Day 1 batches independent request states. Day 2
splits long prefills so they cannot monopolize the scheduler. Day 3 replaces a
growing dense cache with fixed-size pages while retaining a dense-gather
compatibility path. Day 4 removes that gather by teaching attention to walk the
page table directly with a correctness-first schedule. Day 5 then tiles that
same page-walking operation with Week 2's matrix fragments. Page translation
is therefore introduced before it is optimized.

The final model runs paged FlashAttention for long prefill and the paged vector
kernel for short queries. Both schedules read the same page pool through the
same block-table interface; neither rebuilds dense K/V.

Paged attention is not an automatic single-request latency win. It primarily
improves serving capacity, cache reuse, and batching; page-table indirection
can make one request slower. This week measures those tradeoffs rather than
assuming an algorithm with a production name is already production-fast. The
[performance appendix](./appendix-performance.md) keeps kernel throughput and
serving usability as separate measurements.

Week 4 uses a tool-calling coding agent as a demanding multi-turn workload for
this serving substrate. The agent repeatedly changes a long prompt by a small
suffix, pauses while a tool runs, branches after checkpoints, and resumes after
process exit. Supporting that workload will require new session and cache
contracts without changing the attention math built this week.

{{#include copyright.md}}
