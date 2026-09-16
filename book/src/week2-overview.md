<!--
  tiny-llm-book © 2022-2026 by Alex Chi Z is licensed under CC BY-NC-SA 4.0
-->

# 🚧 Week 2: A Step Closer to vLLM

Week 1 leaves you with a readable Qwen3 model that can generate text. This week,
you will turn it into a measured single-request serving path. After each change,
you will rerun the same synchronized workload, find where time now goes, and use
that evidence to choose what to change next. Instead of collecting unrelated
kernels, you will build an optimization story you can explain.

Days 1–5 form the main route. Day 6 is an optional lab for an operator that
matters to a workload you choose. On Day 7, you will test Split-K where a
short-shape measurement suggests it may help, then make a final keep-or-reject
decision on the fixed product workload.

Begin with [Day 1: KV Cache](./week2-01-kv-cache.md), where you will stop
recomputing the entire prefix for every generated token.

> ⏱️ **Time commitment.** Days 3–5 introduce custom Metal kernels and may take
> substantially longer than Week 1 Days 6–7. You may skip Day 6. On Day 7, your
> Split-K implementation must be correct, but the course does not require an
> absolute speed or a universal crossover point.

Week 2 keeps BF16 for dense weights, quantization scales and biases,
activations, projections, KV-cache entries, and model-facing kernel outputs.
Packed W4 weight codes use `uint32`. Reductions, dot products, and
online-softmax state accumulate in FP32 before returning BF16. Week 3 inherits
these interfaces and precision boundaries.

## Measure, Change, and Measure Again

Use the same four-part loop at each checkpoint:

1. **Check correctness.** Run the focused supplied test before timing.
2. **Describe the workload.** Record model, checkpoint, phase, token counts,
   prefill-logit mode, warmups, iterations, software, and device.
3. **Find the next cost.** Name the dominant operator category and make one
   bounded hypothesis before editing.
4. **Rerun and decide.** Repeat the identical product and attribution workload,
   then record `keep`, `reject`, or `inconclusive`, along with evidence that
   would change your conclusion.

You can complete this loop with the synchronized benchmark and portable
attribution runner. Apple GPU capture and `gpudebug` appear only in the
[optional macOS 27 lab](./week2-advanced-profiling.md); neither is a
prerequisite.

## Daily Checkpoints

1. **KV cache:** make decode incremental and compare it with the Week 1 model on
   a matched workload.
2. **Discover:** learn to synchronize a measurement, attribute the cached
   model, and choose one bounded optimization. In the checked run, dense
   projections became the next target.
3. **Packed W4 matvec:** keep weights packed while you optimize decode
   projections, then re-profile. In the checked run, normalization, position,
   and activation work became visible next.
4. **Fused model kernels:** implement RMSNorm, RoPE, and SwiGLU one at a time.
   Keep each change only after a matched measurement, then re-profile prefill
   before choosing Day 5.
5. **SIMD-matrix prefill:** replace the matrix-shaped projection schedule chosen
   from the fixed 128-token prefill attribution.
6. **Optional operator lab:** choose a secondary operator category for one
   explicit workload. The supplied branch studies bounded decode attention,
   but an equivalent evidence-led operator experiment is also valid.
7. **Conditional Split-K and final decision:** try an under-filled 32-token
   projection while preserving the unsplit Day 5 fallback. Finish by rerunning
   the fixed 128×129 product workload and deciding whether to keep the change.

## What Is Supplied and What You Own

The starter gives you model loading, the extension build system, benchmark and
attribution runners, correctness tests, Python reference equations, stable
checkpoint interfaces, and a compact checked M4 Pro evidence file. You will
build the cache transition, integrate packed weights, implement the custom
operators, and turn each measurement into a decision record.

The completed course path uses your implementations, not MLX replacements, for
the operators it asks you to build. If you want to reach the later serving
mechanisms without implementing one custom kernel, keep the course interface
and connect the corresponding MLX operator locally. This substitution stays
inside your course model. It is different from `--solution mlx`, which runs the
separate full-MLX model.

## Check Your Progress

Run the canonical selector after each day:

| Course day | Test command |
|---|---|
| Day 1 | `pdm run test --week 2 --day 1` |
| Day 2 | `pdm run test --week 2 --day 2` |
| Day 3 | `pdm run test --week 2 --day 3` |
| Day 4 | `pdm run test --week 2 --day 4` |
| Day 5 | `pdm run test --week 2 --day 5` |
| Day 6 (optional) | `pdm run test --week 2 --day 6` |
| Day 7 | `pdm run test --week 2 --day 7` |

When a command runs a model, benchmark, profile, capture, or reducer, pass
`--solution tiny_llm` exactly as the chapter shows. Some command-line tools
otherwise default to the completed reference, so omitting it may measure code
you did not write.

### Bring Forward Work from the Earlier Day Order

An earlier course order put decode attention on Day 5 and SIMD-matrix prefill on
Day 6. If your checkout contains work from that order, you can keep it. First
complete the current Day 5 SIMD gate, then use the optional Day 6 gate to check
your retained attention implementation. The ordinary `--week 2 --day 5` and
`--week 2 --day 6` commands above are the only selectors you need. Old Day 5
and Day 6 bookmarks now redirect to the corresponding canonical lessons.

## What the Gates Check

Most required gates check public behavior: checkpoint and workload identity,
operator results, fallbacks, synchronized output, and the decision-record
schema. You may organize most internals differently. Course-ownership and
extension-integration witnesses intentionally preserve explicit source and
header seams. The gates do not grade device timings or require the optional
`gpudebug` tooling.

Read the checked example as one machine's optimization story, not a portable
speed claim. Its absolute measurements come from one M4 Pro running macOS 27
with Qwen3-4B, a fixed 128-token prompt and 129-output-token product control,
and `n=2` balanced product samples. Six of eight captures exposed full
shader/counter detail. The pre-SIMD prefill capture did not expose a shader
ranking, while the Split-K capture exposed only static dispatch. The example
marks the missing data unavailable instead of guessing.

## Continue to Week 3

By the end of Week 2, your model decodes one token at a time from a dense KV
cache, chooses separate prefill and decode projection schedules, and keeps its
weights quantized. Week 3 keeps these model, cache, precision, and operator
interfaces while adding paging and batching. You do not need the optional Day
6 attention branch to continue, and Day 7 begins from Day 5's unsplit SIMD path.

The [performance evidence ledger](./appendix-performance.md) shows the checked
causal example and its limits.

{{#include copyright.md}}
