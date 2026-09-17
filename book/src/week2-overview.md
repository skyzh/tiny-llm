<!--
  tiny-llm-book © 2022-2026 by Alex Chi Z is licensed under CC BY-NC-SA 4.0
-->

# 🚧 Week 2: A Step Closer to vLLM

Week 1 leaves you with a readable Qwen3 model that can generate text. This week,
you will turn it into a measured single-request serving path. After each change,
you will rerun a matched workload, find where time now goes, and use that
evidence to choose what to change next. Instead of collecting unrelated
kernels, you will build an optimization story you can explain.

Days 1–5 build the cache and projection path. Day 6 is an optional long-context
dense-attention experiment. Day 7 fuses the two packed W4 MLP projections that
share the same input. Both final chapters begin with a hypothesis and end with a
keep-or-reject gate; neither promises that a new kernel must win.

Begin with [Day 1: KV Cache](./week2-01-kv-cache.md), where you will stop
recomputing the entire prefix for every generated token.

> ⏱️ **Time commitment.** Days 3–5 introduce custom Metal kernels and may take
> substantially longer than Week 1 Days 6–7. You may skip Day 6. On Days 6–7,
> correctness and a matched decision are required; an absolute speedup is not.

Week 2 keeps BF16 for dense weights, quantization scales and biases,
activations, projections, KV-cache entries, and model-facing kernel outputs.
Packed W4 weight codes use `uint32`. Reductions, dot products, and
online-softmax state accumulate in FP32 before returning BF16. Week 3 inherits
these interfaces and precision boundaries.

## Measure the Whole Request First

The fixed workload generates exactly 128 tokens. Token 1 is selected during
prefill, so time to first token (TTFT) includes prompt processing and that first
selection. Time per output token (TPOT) is the median over the remaining 127
decode intervals.

One checked full-MLX baseline on an Apple M4 Pro shows why a 128-token prompt is
a regression control, not the whole product story:

| Prompt tokens | Prefill | Decode | TTFT | TPOT |
|---:|---:|---:|---:|---:|
| 128 | 829.50 tok/s | 88.52 tok/s | 154 ms | 11.298 ms |
| 512 | 854.69 tok/s | 86.07 tok/s | 599 ms | 11.618 ms |
| 2,048 | 727.19 tok/s | 75.34 tok/s | 2.816 s | 13.272 ms |
| 8,192 | 624.71 tok/s | 54.39 tok/s | 13.113 s | 18.386 ms |
| 32,640 | 370.73 tok/s | 25.71 tok/s | 88.044 s | 38.888 ms |

The 32,640-token prompt plus 128 generated tokens reaches the model's native
32,768-token endpoint exactly. A 32,768-token prompt leaves no room to generate
and is therefore a prefill-only point. Treat 65K or 131K as YaRN-qualified or
synthetic stress, not as the native product workload.

Use the same four-part loop at each checkpoint:

1. **Check correctness.** Run the focused supplied test before timing.
2. **Describe the workload.** Record model, checkpoint, phase, prompt and output
   lengths, prefill-logit mode, warmups, iterations, software, and device.
3. **Find the next cost.** Name the dominant model component and make one
   bounded hypothesis before editing.
4. **Rerun and decide.** Repeat the identical product and attribution workload,
   then record `keep`, `reject`, or `inconclusive`, plus the observation that
   would reverse the decision.

You can complete this loop with the synchronized benchmark and portable
attribution runner. Apple GPU capture and `gpudebug` appear only in the
[optional macOS 27 lab](./week2-advanced-profiling.md); neither is a
prerequisite.

## Read Components, Not One Blended Bucket

The earlier checked story combined attention projections and MLP projections
into a single “projections” bucket. That aggregate hid the context crossover.
The replacement replay groups work by model responsibility:

- QKV and output projections with attention score, softmax, and value
  accumulation;
- MLP gate, up, and down projections with SwiGLU;
- normalization, embedding, output head, and residual/framework overhead where
  the replay can measure them.

At the two endpoints, normalized isolated replay shares change as follows:

| Phase | Component | Prompt 128 | Prompt 32,640 |
|---|---|---:|---:|
| Prefill | QKVO + attention core | 29.02% | 66.24% |
| Prefill | MLP projections + SwiGLU | 63.14% | 32.74% |
| Decode | QKVO + attention core | 36.24% | 62.14% |
| Decode | MLP projections + SwiGLU | 43.97% | 26.10% |

These are normalized shares from isolated replays. They are not additive to
the full-model table and are not production or fleet percentages. Their useful
claim is narrower: MLP work dominates more of the short-prompt replay, while
attention grows with context and becomes the larger target near the native
endpoint. That observation motivates different experiments on Days 6 and 7.

## Daily Checkpoints

1. **KV cache:** make decode incremental and compare it with the Week 1 model on
   a matched workload.
2. **Discover:** learn to synchronize a measurement, attribute the cached
   model, and choose one bounded optimization.
3. **Packed W4 matvec:** keep weights packed while you optimize decode
   projections, then re-profile.
4. **Fused model kernels:** implement RMSNorm, RoPE, and SwiGLU one at a time.
   Keep each change only after a matched measurement.
5. **SIMD-matrix prefill:** use a cooperative schedule for matrix-shaped packed
   W4 projections while retaining the short-row path.
6. **Long-context dense-KV attention (optional):** test an online-softmax
   attention path at 128, 512, 2K, and 8K context, with the readable dense path
   as the exact fallback.
7. **Fused gate+up and SwiGLU:** fuse the two packed W4 MLP projections that read
   the same activation, keep the down projection unchanged, and decide from a
   matched row sweep and product controls.

## What Is Supplied and What You Own

The starter gives you model loading, the extension build system, benchmark and
attribution runners, correctness tests, Python reference equations, stable
checkpoint interfaces, and compact checked evidence. You will build the cache
transition, integrate packed weights, implement the custom operators, and turn
each measurement into a decision record.

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

The replacement selectors `long-context-attention` and `fused-gate-up`, their
learner-owned interfaces, and their matched matrix runner are
integration-pending. Until your checkout lists them in public `--help`, use the
daily test command to track progress and do not substitute the retired
`decode-attention` or `split-k` checkpoints. The chapters state the intended
workloads and acceptance rules without inventing output from an interface that
is not present yet.

## Verification Status

Most required gates check public behavior: checkpoint and workload identity,
operator results, fallbacks, synchronized output, selection counters, disable
controls, and the decision-record schema. You may organize most internals
differently. Course-ownership and extension-integration witnesses intentionally
preserve explicit source and header seams. The gates do not grade an absolute
device timing or require the optional `gpudebug` tooling.

The two final performance gates are deliberately pending:

- Day 6 needs at least 5% lower median TPOT at 8K, the same direction in three
  of four matched context pairs, no more than 2% regression at 128 and 512, and
  removal of the gain when the selector is disabled.
- Day 7 needs at least 5% targeted-phase improvement at 512 or 2K rows, the same
  direction in three of four matched pairs, no more than 2% regression at 8K
  or decode, and removal of the gain when the selector is disabled.

Passing correctness does not imply either performance decision. Measure your
implementation and record the result.

## Continue to Week 3

By the end of Week 2, your model decodes one token at a time from a dense KV
cache, chooses separate prefill and decode projection schedules, and keeps its
weights quantized. Week 3 keeps these model, cache, precision, and operator
interfaces while adding paging and batching. Day 6 deliberately stops at
contiguous dense K/V; paged KV, chunking, and paged attention belong to Week 3.
Day 7 starts from Day 5 and does not depend on the optional Day 6 branch.

The [performance evidence ledger](./appendix-performance.md) records the
checked baselines, isolated component shares, decision gates, and limits.

{{#include copyright.md}}
