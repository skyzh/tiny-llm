# COW / Radix Cache Extension Plan (design only)

Status: **design only — not implemented.** This document preserves the deferred
copy-on-write / radix-cache plan so it can be built as a Week 5 module or a
Week 4 extension after the core Week 4 refsol is reviewed. It intentionally
implements nothing; the core Week 4 stack uses sequential rewind/do-again
(feature 5) and covers every learner scenario without concurrent forks.

## Why this is deferred

The core Week 4 thesis is that four planes must agree: durable events/evidence,
model-visible context, derived KV state, and world/approval authority. The
sequential model (checkpoint -> act -> rewind -> do-again) teaches the derived
KV-state plane honestly: reuse the unchanged prefix, recompute only the
divergent suffix. Concurrent COW forks and cross-session prefix sharing are a
*scaling* mechanism, not a correctness one — they answer "how does a production
server serve many sessions/subagents cheaply?" and fit the Week-5-style
optimization arc (like quantization, flash attention, and paged attention did
for earlier weeks).

## What COW would add (capabilities the sequential model cannot provide)

1. **Concurrent divergence** — two live branches sharing immutable KV pages
   while both continue decoding (parallel what-if exploration, concurrent
   subagents). Sequential rewind covers "try A, reverse, try B"; COW covers
   "run A and B at the same time."
2. **Cross-session prefix sharing** — a radix/prefix registry that shares one
   physical copy of a common prompt prefix across many sessions, with
   refcounted immutable pages and copy-on-write tails.

## Design sketch (for a future implementer)

### Page model

- Immutable full pages, content-addressed by exact token block.
- Per-page reference counts; a page is freed only when its last reference
  releases it.
- Copy-on-write partial tail: divergence allocates a private page; the shared
  prefix pages are never mutated.

### Registry API (proposed)

```text
PrefixRegistry(model_hash, tokenizer_hash, block_size, page_budget)
  publish(token_ids) -> page_ids            # idempotent, content-addressed
  acquire(token_ids) -> ForkedPrefix        # longest published block-aligned prefix
  release(page_ids)                         # refcounted, fails on double-free
  stats() -> PrefixStats                    # live/shared/private/refs/budget
ForkedPrefix
  fork(boundary) -> ForkedPrefix            # block-aligned COW child
  append(token_ids) -> ForkedPrefix         # private divergent tail
  close()                                   # releases shared refs exactly once
```

### Safety invariants

- A cache hit requires exact model, tokenizer, and token identity; registry is
  model-scoped and fails closed on identity mismatch.
- Pages are immutable after publish; closing a child can never alter another
  fork's cache.
- Reference counts cannot leak or double-free; a memory budget evicts only
  unreferenced pages.
- Child policies can only narrow parent authority; private child material is
  never placed in a shared prefix.
- Branch summaries stay event-tree-level (feature 3) and need no COW.

### Honest accounting

Report cold vs optimized wall time separately, exact reused/rewound/prefilled/
generated/discarded token counts, metadata-copy and KV-page-copy bytes,
live/shared/private page counts and peak KV bytes, and branch-discard costs.
Never call reduced visible bytes or more cache hits a speedup unless end-to-end
latency improves without changing the accepted action or losing evidence.

### Evaluation

A read-only fake workspace forks two action plans from one prefix, scores a
deterministic expected result, discards one, and proves no external mutation.
Negative tests attempt a write, change the workspace version, and exhaust the
page budget. Real-model runs compare aggregate time-to-first-token and peak KV
bytes after a warm-up publisher.

## Source influence

The design direction is informed by the Week 4 research lanes (Oracle #196,
Tuner #197) and by production radix-cache systems; the core Week 4 stack keeps
the four-plane thesis without this mechanism. No COW behavior is implied in any
core Week 4 feature.
