<!--
  tiny-llm-book © 2022-2026 by Alex Chi Z is licensed under CC BY-NC-SA 4.0
-->

# 🚧 Week 2: A Step Closer to vLLM

You begin Week 2 with the runnable Week 1 Python `mlx.core` Qwen model. Keep
that model intact. Your work lives in a separate Qwen3 path that first changes
the generation algorithm—prefill once, retain a dense KV cache, and decode one
new token at a time—then replaces the operators that dominate that cached
path.

> ⏱️ **Time commitment.** Days 3–7 write and tune custom Metal kernels.
> Completing the full Week 2 sequence typically takes substantially longer than
> Week 1 Days 6–7. All seven days are required core material; plan accordingly.

Week 2 keeps BF16 for dense weights, quantization scales and biases,
activations, projections, KV-cache entries, and model-facing kernel outputs.
Packed W4 weight codes are stored as `uint32`. Numerically sensitive reductions,
dot products, and online-softmax state accumulate in FP32 inside Python
reference expressions or kernel registers. This contract remains in force for
Week 3.

## What You Build

The seven days form one cumulative single-request path. The starter supplies
model loading, the extension build system, benchmark runners, correctness
tests, Python reference equations, and the stable interfaces between
checkpoints. You implement the state transition on Day 1, establish the
measurement control on Day 2, and own the operator work on Days 3–7:

| Learner-owned work | Supplied infrastructure | Optional work |
|---|---|---|
| Days 1–7: cached model integration, matched benchmarking, quantization, fused model kernels, bounded decode-attention, SIMD-matrix prefill, and shape-aware Split-K | Model loading, extension build system, benchmark runners, correctness tests, and Python-reference implementations | The short profiling notice, schedule searches, hardware-specific retuning, and the fixed-workload 80%-of-MLX stretch target |

Run the supplied test selector after each day. Then run the live model or
benchmark command beside that day so an isolated kernel never counts as a
finished checkpoint. The full campaigns, raw samples, rejected experiments,
and retained reference schedules live in the
[performance appendix](./appendix-performance.md); you do not need to recreate
that evidence ledger to complete the exercises.

## The Cumulative Path

- A dense per-request key-value cache for incremental decoding
- Synchronized benchmarking and the dense decode roofline
- Packed W4 quantization and a SIMD matrix-vector Metal kernel
- Fused RMSNorm, RoPE, and SwiGLU Metal kernels
- An online-softmax decode-attention kernel
- A BF16 SIMD-matrix quantized prefill kernel
- A shape-aware split-K schedule for small Qwen prefill matrices
- A last-token output interface for generation
- An optional stretch target of 80% of MLX prefill and decode throughput on one
  fixed Qwen3-4B workload: 128 prompt tokens, 129 output tokens, last-row
  logits, two warmups, and four balanced fresh-process samples. On the checked
  M4 Pro, the final checkpoint reaches 88.2% of full-MLX prefill and 87.0% of
  full-MLX decode throughput. This is not a cross-shape or cross-device target.

The completed Week 2 solution does **not** call MLX-provided implementations of
the operators it teaches. It implements quantized matmul, decode attention,
RMSNorm, RoPE, and SwiGLU in its own Python, C++, or Metal code. In particular,
the required checkpoint does not use `mx.quantized_matmul`, `mx.dequantize`,
`mx.fast` operators, or `mx.fast.scaled_dot_product_attention` as shortcuts.
Its matrix path also avoids `mlx::steel`: the course scaffold leaves the
cooperative tile loader and direct Metal `simdgroup_matrix` fragment
bookkeeping for you to complete. The Day 1 baseline still uses Week 1's Python
`mx.dequantize` loading helper; Day 3 replaces that loading path as part of
keeping weights packed.

If you want to study the serving path without implementing every custom
operator, Days 3–7 each name a local MLX substitute. Keep the same course
interface and substitute only that day's operator. This is different from
`--solution mlx`, which runs the separate complete MLX model and bypasses the
course-owned model, cache, and scheduler.

Week 2 uses `mlx_lm` to load model weights and `mlx.core` for arrays, graph
evaluation, and device synchronization.

## Daily Checkpoints

1. **KV cache:** port the Week 1 operators into a Week 2 model, add
   request-scoped state, and stop recomputing the prefix.
2. **Benchmarking and profiling:** measure the cached model against MLX with a
   matched, synchronized protocol. Profiling is optional and deferred until the
   macOS 27 tooling is available.
3. **Quantize the model:** keep W4 weights packed, implement the matrix-vector
   Metal path, wire it into the live model, and rerun the Day 2 benchmark.
4. **Fused model kernels:** fuse RMSNorm, RoPE, and SwiGLU one operator at a
   time after packed projections narrow the benchmark gap.
5. **Decode attention:** introduce online softmax over its tested
   short-context range and verify it with a matched workload.
6. **SIMD-matrix prefill:** return to the fixed 128-token workload and replace
   the correctness-first matrix path with cooperative tiles.
7. **Split-K prefill:** partition the reduction dimension only for under-filled
   short projections and fall back to Day 6 at the measured crossover.

### Run the Supplied Test Gates

The seven learner days now map one-to-one to the existing supplied selectors:

| Course day | Test command selector |
|---|---|
| Day 1 | `--week 2 --day 1` |
| Day 2 | `--week 2 --day 2` |
| Day 3 | `--week 2 --day 3` |
| Day 4 | `--week 2 --day 4` |
| Day 5 | `--week 2 --day 5` |
| Day 6 | `--week 2 --day 6` |
| Day 7 | `--week 2 --day 7` |

Use the selector beside each chapter while you work; run the complete day gate
before carrying its result forward.

## Week 2 to Week 3

The completed Week 2 model decodes one token at a time from a dense KV cache,
dispatches separate prefill and decode matrix schedules, and keeps weights
quantized throughout. Week 1 continues to use its Python `mlx.core` full-prefix
generation loop.

Week 3 keeps these Week 2 interfaces, but it deliberately changes projection
ownership: canonical dense Week 3 and the Week 3 scheduler factory select MLX
quantized projections. Cache management, attention, paging, batching, and
scheduling remain course-owned. Full MLX is a separate benchmark baseline,
not another name for this hybrid Week 3 course path.

The explicit projection seam is a teaching boundary, not a performance credit
for paging. The performance appendix isolates the seam while holding the
course-owned serving mechanisms fixed, then reports representative absolute
performance after that choice. Keep those two questions separate.

Run `pdm run bench-week2-progression` to measure each checkpoint against the
Week 1 baseline and MLX. Full methodology and cumulative results are in the
[performance appendix](./appendix-performance.md). The default runs reference
checkpoints; add `--solution tiny_llm` to measure your implementation.

{{#include copyright.md}}
