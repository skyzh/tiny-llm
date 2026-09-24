<!--
  tiny-llm-book © 2022-2026 by Alex Chi Z is licensed under CC BY-NC-SA 4.0
-->

# 🚧 Week 2: A Faster Single Request

Week 1 leaves you with a readable Qwen3 model that regenerates from the full
prefix. The **current Week 2 route contains Days 1–3**: reuse previous keys
and values, bound their dense storage, keep projection weights packed, then
reuse W4 and activation tiles during matrix-shaped prefill. Its four
cumulative checkpoints are `kv-cache`, `capacity-cache`, `quantized-matvec`,
and `simd-matmul`.

Begin with [Day 1: Cache and Measure](./week2-01-kv-cache.md). Its first
feedback loop builds the extension required for test collection, runs the
focused KV test, and sends only the new token during decode. Its second loop
bounds storage without exposing unused capacity to attention. After both
checkpoints, compare the same request across Week 1, `kv-cache`, and
`capacity-cache`; keep the serving-only comparison separate from the
all-logit algorithm comparison.

Continue with [Day 2: Keep W4 Packed](./week2-02-quantize-model.md). Start
from a working `capacity-cache` model. Implement selected-row embedding
dequantization, a checked quantized operator, a readable Metal control and
decode matvec, then wire packed projections into the cached model. The Day 2
checkpoint is complete when the model actually calls the packed-weight path.

Continue to [Day 3: SIMD Matrix Prefill](./week2-03-simd-matrix-prefill.md).
Keep Day 2's `quantized-matvec` checkpoint as the pre-edit control. Build a
cooperative matrix kernel for the larger activation shapes, verify partial
output tiles against the readable control, then compare the two checkpoints
under the same cached-model workload.

## Day 1 route

| Step | What you own | Feedback |
|---|---|---|
| Prepare | Build the Week 2 extension after [Week 1 Day 7](./week1-07-sampling-prepare.md) | `pdm run build-ext` |
| Cache the prefix | Implement dense K/V reuse in the model and generation loop | `pdm run test --week 2 --day 1 -- -k 'not capacity'` |
| Bound the cache | Allocate from the request limit, expose only the logical prefix, and preserve reset/rewind/overflow behavior | `pdm run test --week 2 --day 1` after the capacity work |
| Measure | Keep the workload and prefill-logit mode matched | [Day 1 measurement loop](./week2-01-kv-cache.md#measure-the-first-cache-change) |

## Day 2 route

| Step | What you own | Feedback |
|---|---|---|
| Prepare | Keep the Day 1 `capacity-cache` control; build learner and reference extensions for the native Day 2 checks | `pdm run build-ext` and `pdm run build-ext-ref` |
| Keep W4 packed | Dequantize selected embedding rows, validate the wrapper, and implement the Metal matrix control and SIMD matvec | Focused [Day 2 tests](./week2-02-quantize-model.md) |
| Integrate | Route the cached model's projections and output head through the packed operator | Complete Day 2 gate and a live `quantized-matvec` run |
| Measure | Compare capacity, packed W4, and MLX with one model and one workload | [Day 2 measurement loop](./week2-02-quantize-model.md#verify-quantization-in-the-complete-model) |

## Day 3 route

| Step | What you own | Feedback |
|---|---|---|
| Prepare | Complete Day 2, build both native extensions, and save a `quantized-matvec` prefill control | [Day 3 baseline](./week2-03-simd-matrix-prefill.md#keep-the-pre-edit-control) |
| Build the tile | Load BF16 activation and reconstructed W4 fragments cooperatively; accumulate in FP32 and guard partial outputs | Focused [partial-tile test](./week2-03-simd-matrix-prefill.md#task-1-load-and-multiply-a-w4-tile) |
| Integrate | Keep the decode-shaped matvec, dispatch larger matrices to SIMD, and wire `simd-matmul` through the cached model | Complete Day 3 test and live model checkpoint |
| Measure | Compare old and new prefill paths with the same cached 0.6B model and workload | [Day 3 product loop](./week2-03-simd-matrix-prefill.md#measure-the-matched-product) |

The Day 1 starter supplies model loading, test entrypoints, and benchmark and
attribution helpers; you own the cache state and serving loop. Day 2 adds the
packed-weight container and native operator boundaries, but you implement the
embedding, kernels, and model wiring. Day 3 adds the SIMD matrix path behind
that packed operator. The reference solution and full MLX
model are separate controls; they do not fill your learner TODOs. A cache
counter shows which bytes moved, while a synchronized complete-request
comparison shows whether a mechanism helped the chosen workload.

## Later lessons

The reviewed five-day design continues after SIMD matrix prefill with model
primitives and tiled dense prefill attention. Those **Day 4–5 checkpoints are
planned, not shipped in this three-day route**. Their commands and selectors
are not current gates. Week 3 uses later Week 2 interfaces; Days 1–3 alone do
not supply every prerequisite for its learner exercises.

The earlier seven-day book remains available at its old addresses as
[historical Week 2 material](./week2-02-benchmark-profile.md). It preserves
benchmark method, W4 derivation, Apple M1–M4 bandwidth and roofline
calculations, fusion/SIMD mechanisms, optional capture, and the old
bounded-decode and Split-K experiments. Its
[operator-attribution diagram](./week2-kernel-profile.svg) and
[decision diagram](./week2-performance-summary.svg) are also historical
evidence, not diagrams of the current checkout. Those pages describe a different
checkpoint order and may show commands unavailable in this partial branch.
Use Days 1–3 above for the current learner workflow. The
[performance evidence ledger](./appendix-performance.md) is likewise
historical context, not a performance claim for this checkout.

{{#include copyright.md}}
