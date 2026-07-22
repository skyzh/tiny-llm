# 🚧 Week 3 Day 8 (Optional): Speculative Decoding

> 🚧 This optional chapter remains a work in progress.

Speculative decoding uses a smaller draft model to propose several tokens, then
asks the target model to verify them in one call. Accepted draft tokens reduce
the number of target-model decode steps without changing the target
distribution.

The serving runtime needs three additions:

1. a draft-token loop with a bounded proposal length;
2. a verification pass that returns logits for every proposed position;
3. cache rewind when a suffix is rejected.

`TinyKvCache.rewind` is the important interface boundary. Dense caches slice
their suffix; paged caches also return newly unused pages to the pool. Keep this
feature optional: its speedup depends on draft quality, target cost, and the
overhead of verification.

{{#include copyright.md}}
