<!--
  tiny-llm-book © 2022-2026 by Alex Chi Z is licensed under CC BY-NC-SA 4.0
-->

# 🚧 Week 2: A Step Closer to vLLM

> **Status: Experimental.** Week 2 is under active development. Each chapter
> carries its own verification notes; the summary below records what is
> continuously tested versus what is one-machine research evidence.

Week 2 keeps the Week 1 Python `mlx.core` model intact and builds a separate
optimized Qwen3 path for single-request decoding. It begins with the algorithm
change: prefill once, retain a dense KV cache, and decode one new token at a
time. Later chapters introduce kernels that address the costs the KV cache
exposes.

> ⏱️ **Time commitment.** Days 3–7 write and tune custom Metal kernels.
> Completing the full Week 2 sequence typically takes substantially longer than
> Week 1 Days 6–7. All seven days are required core material; plan accordingly.

Week 2 keeps BF16 for dense weights, quantization scales and biases,
activations, projections, KV-cache entries, and model-facing kernel outputs.
Packed W4 weight codes are stored as `uint32`. Numerically sensitive reductions,
dot products, and online-softmax state accumulate in FP32 inside Python
reference expressions or kernel registers. This contract remains in force for
Week 3.

## Verification Status

Reference correctness and decode-attention boundaries are continuously tested
on ARM64 macOS CI with Qwen3-0.6B. Qwen3-4B performance evidence is one-machine
research data measured on an M4 Pro, not a cross-device guarantee. Raw
measurements, rejected experiments, and retained dispatch choices live in the
[performance evidence ledger](./appendix-performance.md).

## Complete the Core Path

The full reference solution is a substantial performance-engineering project.
All seven days are required core material; schedule more than one week if
needed:

| Required work | Provided infrastructure | Optional work |
|---|---|---|
| Days 1–7: cached model integration, matched benchmarking, quantization, fused model kernels, bounded decode-attention, SIMD-matrix prefill, and shape-aware Split-K | Model loading, extension build system, benchmark runners, correctness tests, and Python-reference implementations | The short profiling notice, schedule searches, hardware-specific retuning, and the 80%-of-MLX stretch target |

The tests define API and correctness contracts; they do not require a student
to rediscover the reference schedule. Metal capture, Xcode visualization,
`gpudebug`, and profiling microbenchmarks are not current requirements and are
not acceptance gates.

## What We Will Cover

- A dense per-request key-value cache for incremental decoding
- Synchronized benchmarking and the dense decode roofline
- Packed W4 quantization and a SIMD matrix-vector Metal kernel
- Fused RMSNorm, RoPE, and SwiGLU Metal kernels
- An online-softmax decode-attention kernel
- A BF16 SIMD-matrix quantized prefill kernel
- A shape-aware split-K schedule for small Qwen prefill matrices
- A last-token output interface for generation
- An optional stretch target of 80% of MLX prefill and decode throughput on the
  fixed Week 2 checkpoint

Week 2 does **not** call MLX-provided implementations of the operators we are
learning. Your solution implements quantized matmul, decode attention, RMSNorm,
RoPE, and SwiGLU in its own Python, C++, or Metal code. In particular, the
completed checkpoint does not use `mx.quantized_matmul`, `mx.dequantize`,
`mx.fast` operators, or `mx.fast.scaled_dot_product_attention` as shortcuts.
The Day 1 baseline still uses Week 1's Python `mx.dequantize` loading helper;
Day 3 replaces that loading path as part of keeping weights packed.

Week 2 uses `mlx_lm` to load model weights and `mlx.core` for arrays, graph
evaluation, and device synchronization.

## Weekly Checkpoints

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

Run every group assigned to the chapter before continuing. The supplied test
filenames and selectors are stable historical machine identities; the chapter
headings and navigation now use the same seven-day sequence.

## Week 2 to Week 3

The completed Week 2 model decodes one token at a time from a dense KV cache,
dispatches separate prefill and decode matrix schedules, and keeps weights
quantized throughout. Week 1 continues to use its Python `mlx.core` full-prefix
generation loop.

Week 3 imports these Week 2 interfaces rather than copying or replacing them.
It adds page-table translation and combines Week 2's online softmax,
SIMD-matrix tiling, and page walking in one paged FlashAttention operator. That
boundary lets each week's model remain understandable and runnable on its own.

Run `pdm run bench-week2-progression` to measure each checkpoint against the
Week 1 baseline and MLX. Full methodology and cumulative results are in the
[performance appendix](./appendix-performance.md). The default runs reference
checkpoints; add `--solution tiny_llm` to measure your implementation.

{{#include copyright.md}}
