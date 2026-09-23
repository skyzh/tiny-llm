<!--
  tiny-llm-book © 2022-2026 by Alex Chi Z is licensed under CC BY-NC-SA 4.0
-->

# 🚧 Week 2: A Faster Single Request

Week 1 leaves you with a readable Qwen3 model that can generate text. Week 2
turns it into a measured single-request path. Start with the waste you can see:
without a cache, every generated token recomputes the entire prefix. After
caching removes that work, a matched measurement can tell you which operator
or data movement is worth changing next. Five lessons build one cumulative
model; every lesson keeps a readable control for comparison.

Begin with [Day 1](./week2-01-kv-cache.md). You will first send only the new
token during decode, then remove the cache's repeated prefix copies without
exposing unused capacity to attention.

> ⏱️ **Time commitment.** Days 2–5 include C++ and Metal kernels and may take
> substantially longer than Week 1. Use the documented operator off-ramps if
> you need to continue the serving course while studying a kernel separately.

Week 2 keeps BF16 activations, scales, biases, KV entries, and model-facing
outputs. Packed W4 codes use `uint32`. Reductions, dot products, and
online-softmax state accumulate in FP32 before returning BF16. Week 3 inherits
these interfaces and precision boundaries.

## Measure, Change, Measure Again

Keep one request shape fixed while you change a mechanism:

1. Check the supplied correctness gate and the complete generation path.
2. Record model, prompt/output lengths, checkpoint, prefill-logit mode, device,
   software, warmups, and whether the process was fresh.
3. Attribute the same phase and shape. Predict what one change should remove.
4. Repeat the matched product and component controls. Record what ran, what
   changed, and what result would reverse your choice.

The old checked M4 Pro example began with dense projections dominant in cached
decode. W4 reduced projection traffic; later fused primitives and SIMD matrix
prefill changed the profile again. These are historical observations from one
source tree and machine, not measurements of your checkout. A cache copy
counter, an operator time, and a complete-request median answer different
questions. Do not add their percentages.

![Nine cumulative Week 2 checkpoints followed by separate capacity, RMSNorm, and tiled attention evidence cards. The cards distinguish component results from selected complete-request gains.](./week2-kernel-profile.svg)

The synchronized benchmark and portable attribution runner are enough for the
required learning loop. The [optional macOS capture lab](./week2-advanced-profiling.md)
shows how to inspect a GPU trace when that toolchain is available; a trace is
not a prerequisite.

## Five Days, Nine Checkpoints

| Day | What you change | Public checkpoint after the change |
|---|---|---|
| [1. Cache and measure](./week2-01-kv-cache.md) | Incremental decode, then request-bounded K/V storage; compare the same request and derive the decode roofline | `kv-cache`, `capacity-cache` |
| [2. Packed W4](./week2-02-quantize-model.md) | Packed embedding and projections; decode-shaped SIMD matvec | `quantized-matvec` |
| [3. SIMD matrix prefill](./week2-03-simd-matrix-prefill.md) | Reuse W4 and activation tiles across many prompt rows | `simd-matmul` |
| [4. Model primitives](./week2-04-fused-model-kernels.md) | RMSNorm, RoPE, and SwiGLU, each integrated and checked separately | `rmsnorm`, `rope`, `swiglu` |
| [5. Tiled prefill attention](./week2-05-tiled-prefill-attention.md) | Online-softmax BF16/D128 prefill with a readable attention fallback; run the completed product | `tiled-prefill`, `selected` |

`selected` contains request-bounded capacity, register-cached RMSNorm, and
tiled dense prefill attention alongside the cumulative packed-W4, SIMD,
RoPE, and SwiGLU work. One-token decode attention stays readable. The earlier
bounded decode-attention and Split-K experiments are
[historical context](./appendix-performance.md#retired-week-2-experiments), not
current checkpoints or extra days.

## What Is Supplied and What You Own

The starter gives you model loading, extension build plumbing, benchmark and
attribution runners, Python reference equations, tests, and stable checkpoint
interfaces. You implement the cache transition, packed-weight path, and
course-owned operators. MLX is a baseline and numerical oracle. If you take an
operator off-ramp, keep the course's interface and delegate only that operator
locally; `--solution mlx` runs a different complete model and does not test
your cache or model wiring.

The supplied test tool still identifies the nine checkpoints with its earlier
seven gate IDs. Each chapter shows the gate ID that exercises its checkpoint;
it is a test selector, not an extra lesson. After all five lessons, run the
complete Week 2 gate:

```bash
pdm run test --week 2
```

When a command runs a model, benchmark, profiler, capture, or reducer, pass
`--solution tiny_llm` as shown. Some tools otherwise select the reference
solution. Keep a locally available model and the same request on both sides
of any comparison.

## Evidence and Limits

The [performance appendix](./appendix-performance.md) distinguishes
correctness and dispatch witnesses, component comparisons, complete-request
measurements, historical data, and unavailable rows. The accepted selected
campaign on Qwen3-4B improved total latency versus all mechanisms off by
10.427% at 128/128, 9.996% at 512/128, 15.375% at 2K/16, and 14.193% at
2K/128. Its matched full-MLX throughput ratios were about 0.804, 0.822,
0.829, and 0.769. The 80% direction was met on the first three rows and
missed on 2K/128. The 2K/512 and 8K/128 product rows are unavailable after
environmental contamination. Component gains cannot fill those cells.

![Week 2 evidence ladder: correctness, routing, component comparison, complete-request comparison, then a bounded decision. Missing product rows stay unavailable rather than being inferred from components.](./week2-performance-summary.svg)

## Continue to Week 3

By the end of Week 2, your model decodes one token at a time from a request
cache, keeps projection weights packed, selects separate prefill and decode
schedules, and has readable controls for its optimized operators. Week 3 keeps
these model, cache, precision, and operator interfaces while adding paging and
batching. No Week 2 measurement here establishes a production serving policy.

{{#include copyright.md}}
