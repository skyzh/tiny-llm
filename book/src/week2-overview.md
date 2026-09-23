<!--
  tiny-llm-book © 2022-2026 by Alex Chi Z is licensed under CC BY-NC-SA 4.0
-->

# 🚧 Week 2: A Faster Single Request

Week 1 leaves you with a readable Qwen3 model that regenerates from the full
prefix. The **current Week 2 route contains Day 1 only**: reuse previous keys
and values, then give their dense storage a request bound. The two cumulative
checkpoints are `kv-cache` and `capacity-cache`.

Begin with [Day 1: Cache and Measure](./week2-01-kv-cache.md). Its first
feedback loop builds the extension required for test collection, runs the
focused KV test, and sends only the new token during decode. Its second loop
bounds storage without exposing unused capacity to attention. After both
checkpoints, compare the same request across Week 1, `kv-cache`, and
`capacity-cache`; keep the serving-only comparison separate from the
all-logit algorithm comparison.

## Day 1 route

| Step | What you own | Feedback |
|---|---|---|
| Prepare | Build the Week 2 extension after [Week 1 Day 7](./week1-07-sampling-prepare.md) | `pdm run build-ext` |
| Cache the prefix | Implement dense K/V reuse in the model and generation loop | `pdm run test --week 2 --day 1 -- -k 'not capacity'` |
| Bound the cache | Allocate from the request limit, expose only the logical prefix, and preserve reset/rewind/overflow behavior | `pdm run test --week 2 --day 1` after the capacity work |
| Measure | Keep the workload and prefill-logit mode matched | [Day 1 measurement loop](./week2-01-kv-cache.md#measure-the-first-cache-change) |

The starter supplies model loading, the extension build, test entrypoints,
and benchmark and attribution helpers. You own the cache state, its model
wiring, and the serving loop. The reference solution and full MLX model are
separate controls; they do not fill your learner TODOs. A cache counter shows
which bytes moved, while a synchronized complete-request comparison shows
whether that mechanism helped the chosen workload.

## Later lessons

The reviewed five-day design continues with packed W4, SIMD matrix prefill,
model primitives, and tiled dense prefill attention. Those **Day 2–5
checkpoints are planned, not shipped in this Day 1 route**. Their commands and
selectors are not Day 1 gates. Week 3 uses later Week 2 interfaces; Day 1
alone does not supply every prerequisite for its learner exercises.

The earlier seven-day book remains available at its old addresses as
[historical Week 2 material](./week2-02-benchmark-profile.md). It preserves
benchmark method, W4 derivation, Apple M1–M4 bandwidth and roofline
calculations, fusion/SIMD mechanisms, optional capture, and the old
bounded-decode and Split-K experiments. Its
[operator-attribution diagram](./week2-kernel-profile.svg) and
[decision diagram](./week2-performance-summary.svg) are also historical
evidence, not diagrams of the Day 1 checkout. Those pages describe a different
checkpoint order and may show commands unavailable in this partial branch.
Use Day 1 above for the current learner workflow. The
[performance evidence ledger](./appendix-performance.md) is likewise
historical context, not a performance claim for this checkout.

{{#include copyright.md}}
