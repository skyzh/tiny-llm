# Week 3: Build a Mini vLLM

> **Course status:** KV cache, continuous batching, FlashAttention, and chunked
> prefill come from the original course. Paged attention and the optional
> extensions remain works in progress.

Week 3 takes the optimized single-request model from Week 2 and builds a serving
engine around it. The optimized operator layer remains unchanged; this week is
about cache ownership, scheduling, prefill, and runtime metadata.

## What We’ll Cover

- A dense key-value cache and its lifecycle
- Continuous batching and request-slot reuse
- FlashAttention for prefill
- Chunked prefill and scheduler fairness
- Paged KV storage and page-walking attention
- Optional MoE and speculative-decoding extensions

Week 4 owns application concerns such as RAG and tool calling. This separation
keeps Week 3 focused on the reusable serving substrate.

{{#include copyright.md}}
