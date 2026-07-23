# 🚧 Week 3: Build a Mini vLLM

> 🚧 This overview was substantially revised and is a work in progress.

> **Course status:** The overview and Days 1–5 are under review after this
> restructure. Their topics come from the original course; the performance lab
> and speculative-decoding chapter are new works in progress. The lightly
> adapted MoE chapter retains its completed status.

Week 3 takes the optimized single-request model from Week 2 and builds a serving
engine around it. The Week 2 operator interfaces remain intact; this week adds
opt-in prefill kernels, multi-request cache ownership, scheduling, and runtime
metadata. It also contains a performance lab for optimizations deliberately
left out of the minimal Week 2 checkpoint.

## What We’ll Cover

- Continuous batching and request-slot reuse
- FlashAttention for prefill
- Chunked prefill and scheduler fairness
- Paged KV storage and page-walking attention
- Optional prefill, embedding, and scheduling experiments
- Optional MoE and speculative-decoding extensions

The ordering is intentional: students implement dense FlashAttention on Day 2
before paged storage and paged attention on Days 4–5. The paged prefill kernel
then reuses the same BF16 tiling and online-softmax machinery, leaving page
translation and decode work partitioning as the new concepts.

The final model runs dense FlashAttention directly on fresh K/V for an
ordinary first prefill, stores the same K/V in paged cache, and switches to the
paged vector kernel for decode. Page-walking prefill remains for a chunked
prefill that already has cached history.

On Apple silicon, FlashAttention and paged attention are not automatic latency
wins. FlashAttention has the best opportunity at long prefill lengths, where
avoiding an `L x S` intermediate can save substantial memory traffic. Paged
attention primarily improves serving capacity, cache reuse, and batching; its
page-table indirection can make a single request slower. This week measures
those tradeoffs rather than assuming an algorithm with a production name is
already production-fast.

Week 4 owns application concerns such as RAG and tool calling. This separation
keeps Week 3 focused on the reusable serving substrate.

{{#include copyright.md}}
