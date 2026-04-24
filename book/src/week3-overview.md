# Week 3: Serving

In Week 3 of the course, we move from the "tiny vLLM" baseline to the next layer of serving-system ideas. Week 2 gave us the core runtime loop: KV cache, quantized kernels, FlashAttention, chunked prefill, and continuous batching. Week 3 is where we start addressing the limitations of that baseline and connect the model runtime to more realistic serving features.

## What We’ll Cover

* Paged attention
    * Part 1: paged KV cache and the page-table abstraction
    * Part 2: block tables, paged runtime metadata, and the real attention path
* Additional serving optimizations
    * MoE routing and serving considerations
    * speculative decoding
    * long-context techniques
* Model interaction with the outside world
    * retrieval-augmented generation (RAG)
    * tool calling / agent-style execution

The goal of Week 3 is not just to make the model faster. It is to understand how a serving system evolves once the basic decode loop already works: how memory is managed, how runtime metadata flows into kernels, and how the serving stack coordinates with external systems.

{{#include copyright.md}}
