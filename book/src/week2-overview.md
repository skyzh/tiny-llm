<!--
  tiny-llm-book © 2022-2026 by Alex Chi Z is licensed under CC BY-NC-SA 4.0
-->

# 🚧 Week 2: A Step Closer to vLLM

Week 2 turns the readable Week 1 Qwen3 model into a measured single-request
serving path. You will change one mechanism, run the same synchronized
workload, attribute the remaining cost, and let that evidence choose the next
change. The result is a causal optimization loop rather than a checklist of
kernels.

The core route is Days 1–5. Day 6 is an optional workload-conditioned operator
lab. Day 7 adds Split-K only where a short-shape measurement supports it, then
closes the week with a fixed-workload keep/reject decision.

> ⏱️ **Time commitment.** Days 3–5 implement custom Metal kernels and can take
> substantially longer than Week 1 Days 6–7. Day 6 is optional. Day 7's
> Split-K schedule is conditional: correctness is required, but no absolute
> timing or universal crossover gates completion.

Week 2 keeps BF16 for dense weights, quantization scales and biases,
activations, projections, KV-cache entries, and model-facing kernel outputs.
Packed W4 weight codes use `uint32`. Reductions, dot products, and
online-softmax state accumulate in FP32 before returning BF16. Week 3 inherits
these interfaces and precision boundaries.

## The Causal Loop

Every checkpoint follows the same four moves:

1. **Prove correctness.** Run the focused supplied test before timing.
2. **Freeze the workload.** Record model, checkpoint, phase, token counts,
   prefill-logit mode, warmups, iterations, software, and device.
3. **Attribute before editing.** Name the dominant operator category and one
   bounded hypothesis.
4. **Repeat and decide.** Rerun the identical product and attribution workload,
   then record `keep`, `reject`, or `inconclusive` and a falsifier.

The synchronized benchmark and portable attribution runner are the ordinary
path for every learner. Apple GPU capture and `gpudebug` are an
[optional macOS 27 lab](./week2-advanced-profiling.md), never a prerequisite.

## Daily Checkpoints

1. **KV cache:** make decode incremental, then record a matched Week 1 versus
   cached Week 2 observation.
2. **Discover:** learn synchronized measurement, attribute the cached model,
   and choose the first bounded optimization. The checked run selected dense
   projections.
3. **Packed W4 matvec:** keep weights packed, optimize decode projections, and
   re-profile. The checked run then exposed normalization, position, and
   activation work.
4. **Fused model kernels:** implement RMSNorm, RoPE, and SwiGLU one at a time,
   retaining each from a matched measurement. Re-profile prefill before
   choosing Day 5.
5. **SIMD-matrix prefill:** replace the matrix-shaped projection schedule
   selected by the fixed 128-token prefill attribution.
6. **Optional operator lab:** choose a secondary category for one explicit
   workload. The supplied branch studies bounded decode attention; an
   equivalent evidence-led operator experiment is valid.
7. **Conditional Split-K and final decision:** test an under-filled 32-token
   projection, preserve the unsplit Day 5 fallback, then rerun the fixed
   128×129 product workload and keep or reject the change there.

## What Is Supplied and What You Own

The starter supplies model loading, the extension build system, benchmark and
attribution runners, correctness tests, Python reference equations, stable
checkpoint interfaces, and a compact checked M4 Pro evidence file. You own the
cache transition, packed-weight integration, custom operator behavior, and
the evidence-to-decision record.

The completed course path does not use MLX-provided implementations of the
operators it teaches. If you want to study the later serving mechanisms
without implementing one custom kernel, keep that course interface and wire
the corresponding MLX operator locally. That local substitution is not the
same as `--solution mlx`, which runs the separate full-MLX model.

## Run the Supplied Gates

New learners use the canonical selectors:

| Course day | Test command |
|---|---|
| Day 1 | `pdm run test --week 2 --day 1` |
| Day 2 | `pdm run test --week 2 --day 2` |
| Day 3 | `pdm run test --week 2 --day 3` |
| Day 4 | `pdm run test --week 2 --day 4` |
| Day 5 | `pdm run test --week 2 --day 5` |
| Day 6 (optional) | `pdm run test --week 2 --day 6` |
| Day 7 | `pdm run test --week 2 --day 7` |

Commands that exercise a model, benchmark, profile, capture, or reducer name
`--solution tiny_llm` explicitly. The command-line tools otherwise default to
the completed reference in some contexts, which would not measure your work.

### Migrating Work from the Earlier Day Order

The course formerly taught decode attention on Day 5 and SIMD-matrix prefill
on Day 6. That work remains useful. If your checkout already contains those
solutions, preserve it and run:

```bash
pdm run test --week 2 --day 5 --legacy-week2-order
pdm run test --week 2 --day 6 --legacy-week2-order
```

The saved checkpoints remain available as `legacy-day-5` and
`legacy-day-6`. Complete the new Day 5 SIMD gate, then treat your attention
implementation as the optional Day 6 branch. The old
[Day 5](./week2-05-decode-attention.md) and
[Day 6](./week2-06-simd-matrix-prefill.md) URLs remain as migration pages, so
existing bookmarks still resolve.

## Verification Status

The required gates check public behavior: checkpoint and workload identity,
operator results, fallbacks, synchronized output, and decision-record schema.
They do not grade exact private function names, Metal symbols, file routes,
device timings, or whether optional `gpudebug` tooling is installed. A fresh
or equivalently organized learner solution can pass.

For the checked example, all absolute measurements are bounded to one M4 Pro,
macOS 27, Qwen3-4B, the fixed 128-token prompt and 129-output-token product
control, and `n=2` balanced product samples. Six of eight captures exposed full
shader/counter detail; the pre-SIMD prefill capture did not expose a shader
ranking, and the Split-K capture exposed only static dispatch. Missing data is
reported as unavailable, never inferred.

## Week 2 to Week 3

The completed Week 2 model decodes one token at a time from a dense KV cache,
dispatches separate prefill and decode projection schedules, and keeps weights
quantized. Week 3 preserves these model, cache, precision, and operator
interfaces while adding paging and batching. Day 6's optional attention branch
is not a Week 3 prerequisite; Day 7 likewise starts from Day 5's unsplit SIMD
path.

See the [performance evidence ledger](./appendix-performance.md) for the
checked causal example and its limits.

{{#include copyright.md}}
