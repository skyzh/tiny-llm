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

> ⏱️ **Time commitment.** Days 3–6 write and tune custom Metal kernels.
> Completing the full Week 2 sequence typically takes substantially longer than
> Week 1 Days 6–7. All six days are required core material; plan accordingly.

Week 2 keeps BF16 for dense weights, quantization scales and biases, activations, projections, KV-cache entries, and model-facing kernel outputs. Packed W4 weight codes are stored as `uint32`. Numerically sensitive reductions, dot products, and
online-softmax state accumulate in FP32 inside Python reference expressions or
kernel registers. This contract remains in force for Week 3.

## Verification Status

Reference correctness and decode-attention
boundaries are continuously tested on ARM64 macOS CI with Qwen3-0.6B.
Qwen3-4B performance evidence is one-machine research data measured on an M4
Pro, not a cross-device guarantee. Raw measurements, rejected experiments, and
retained dispatch choices live in the
[performance evidence ledger](./appendix-performance.md).

## Choose a Completion Track

The full reference solution is a small performance-engineering project, not a
reasonable one-week requirement for every student. All six days are required
core material:

| Track | Required work | Provided infrastructure | Optional work |
|---|---|---|---|
| Core course | Days 1–6: cached model integration, benchmarking and quantization, fused model kernels, bounded decode-attention, SIMD-matrix prefill, and shape-aware Split-K | Model loading, extension build system, benchmark/profile runners, correctness tests, and the Python-reference implementations | Xcode counter capture, schedule searches, and hardware-specific retuning |

The tests define API and correctness contracts; they do not require a student
to rediscover the reference schedule. The 80%-of-MLX acceptance target is an
optional stretch goal measured on your own hardware.

## What We Will Cover

- A dense per-request key-value cache for incremental decoding
- Synchronized benchmarking, the decode roofline, and packed W4 quantization
- A SIMD matrix-vector quantized decode kernel implemented in Metal
- Fused RMSNorm, RoPE, and SwiGLU Metal kernels
- An online-softmax decode-attention kernel
- A BF16 SIMD-matrix quantized prefill kernel
- A shape-aware split-K schedule for small Qwen prefill matrices
- A last-token output interface for generation
- An optional stretch target of 80% of MLX prefill and decode
  throughput on the fixed Week 2 checkpoint

Week 2 does **not** call MLX-provided implementations of the operators we are
learning. Your solution implements quantized matmul, decode attention,
RMSNorm, RoPE, and SwiGLU in its own Python, C++, or Metal code. In
particular, the completed checkpoint does not use `mx.quantized_matmul`,
`mx.dequantize`, `mx.fast` operators, or
`mx.fast.scaled_dot_product_attention` as shortcuts. The Day 1 baseline still
uses Week 1's Python `mx.dequantize` loading helper; Day 2 replaces that
loading path as part of keeping weights packed.

Week 2 uses `mlx_lm` to load model weights and `mlx.core` for arrays, graph
evaluation, and device synchronization.

## Weekly Checkpoints

1. **KV cache:** port the Week 1 operators into a Week 2 model, add
   request-scoped state, and stop recomputing the prefix.
2. **Benchmark, profile, and quantize:** measure the cached model against MLX,
   identify projection weight reads as the bottleneck, replace dense 16-bit
   weights with packed W4, and implement the SIMD matrix-vector Metal kernel.
3. **Fused model kernels:** after packed projections narrow the measured gap
   with MLX, profile and fuse RMSNorm, RoPE, and SwiGLU one operator at a
   time.
4. **Decode attention:** profile score computation, softmax, and value
   aggregation across cached context lengths. Introduce online softmax over
   its measured short-context range and verify with a matched workload.
5. **SIMD-matrix prefill:** return to the fixed 128-token workload and its
   prefill profile, where the correctness-first quantized matrix path
   dominates. Introduce 8×8 matrix fragments with FP32 accumulation.
6. **Split-K prefill:** the Day 5 shape sweep reveals under-filled grids at
   short row counts, most clearly in narrow Qwen K/V projections. Partition
   the reduction dimension and fall back to Day 5 at the measured crossover.

At each boundary, end-to-end and operator measurements select the next kernel
family. An optional Xcode trace explains what happens inside a representative
GPU kernel.

## Week 2 to Week 3

The completed Week 2 model decodes one token at a time from a dense KV cache,
dispatches separate prefill and decode matrix schedules, and keeps weights
quantized throughout. Week 1 continues to use its Python `mlx.core` full-prefix
generation loop.

Week 3 imports these Week 2 interfaces rather than copying or replacing them.
It adds page-table translation and combines Week 2's online softmax, SIMD-matrix
tiling, and page walking in one paged FlashAttention operator. That boundary
lets each week's model remain understandable and runnable on its own.

Run `pdm run bench-week2-progression` to measure each checkpoint against the
Week 1 baseline and MLX. Full methodology and cumulative results are in the
[performance appendix](./appendix-performance.md). The default runs reference
checkpoints; add `--solution tiny_llm` to measure your implementation.

{{#include copyright.md}}
