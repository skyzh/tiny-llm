# 🚧 Week 3: Build a Mini vLLM

> 🚧 This overview and chapters carrying the same marker are under review and
> may change.

At the end of Week 2 your model serves one request at a time, and serves it
quickly. A real server has many users at once. Their prompts arrive at
different times, have different lengths, and finish at different times. This
week you turn the model into a small serving engine that keeps several
requests moving together.

By the end of Week 3 your engine:

- decodes several requests in one model call, and gives a finished request's
  place to a waiting one straight away (**continuous batching**, Day 1);
- splits a long prompt into chunks, so one long prefill cannot hold up
  everyone else's next token (**chunked prefill**, Day 2);
- stores each request's keys and values in fixed-size pages from one shared
  pool, instead of one growing buffer per request (**paged KV cache**, Day 3);
- reads those pages directly inside attention, with no copy back into one
  dense buffer (**direct paged attention**, Day 4); and
- runs long prompts through a tiled attention schedule that reads the same
  pages (**paged FlashAttention**, Day 5).

Days 3 to 5 introduce page translation before optimizing it. Day 3 still has a
simple path that gathers pages back into dense tensors, so you can check the
page bookkeeping on its own. Day 4 removes that gather with a straightforward
kernel that walks the page table and can serve every query shape. Day 5 adds a
faster tiled schedule for the common long-prefill case (BF16 prompts), and
keeps Day 4's kernel for short queries and other shapes. Every schedule reads
the same page pool through the same page-table interface.

Two optional days follow. Day 6 adds Mixture-of-Experts model support, which
is independent of the cache and scheduler. Day 7 adds speculative decoding,
which needs to rewind the cache exactly when a guess is rejected, and needs
the page-aware attention path for queries longer than one token to check
several guesses at once.

## One Change to the Model

Week 3 is about the serving system, not the projection kernels. From Day 1 on,
the Week 3 model sends its quantized projections to MLX's own
`mx.quantized_matmul` instead of your Week 2 kernels. Everything else stays
yours: normalization, activation, RoPE, the cache, attention, paging and the
scheduler. Your Week 2 checkpoints keep working unchanged. Day 1 shows the
switch.

Without it, every scheduler and cache measurement in Week 3 would also carry
the remaining cost of the teaching kernels, and that cost would hide the
effects you are trying to see.

## What Paging Buys, and What It Costs

Paging is not a free speedup for a single request. Looking up each page adds
work, so one request on its own can get slower. What paging buys is memory:
storage that grows a page at a time, pages that can be reused as soon as a
request finishes, and room for more requests in the same batch.

An earlier measurement, made before Week 2 was rebuilt, showed those memory
benefits and the single-request cost. It was not made on this checkout, and it
does not show that more requests fit in a fixed amount of memory; that would
need a sweep with the memory capped. Each chapter ends with a measurement you
can run yourself. The [performance appendix](./appendix-performance.md) keeps
the older results and exactly where they came from.

{{#include copyright.md}}
