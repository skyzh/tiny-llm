# 🚧 Week 2 Day 4: Fused Model Kernels

Day 3 leaves the cached model using packed projections and a SIMD matrix
prefill schedule. Day 4 keeps the Week 1
Python equations as readable oracles and completes three separate extension
shells already present in the starter:

```plain
src/tiny_llm/week2_kernels.py
src/extensions/src/week2_kernels.cpp
src/extensions/src/week2_kernels.metal
```

Work in checkpoint order: implement and integrate RMSNorm, then RoPE, then
SwiGLU. After each operator, run its focused test and the live cumulative
checkpoint. This keeps a local operator failure separate from an integration
regression before all three are active:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k rms
pdm run test --week 2 --day 4 -- -k rope
pdm run test --week 2 --day 4 -- -k swiglu
```

RMSNorm, RoPE, and SwiGLU surround the projections in every transformer layer.
Week 1 gives you readable Python `mlx.core` equations; your Week 2 path keeps
their interfaces and supplies purpose-built Metal kernels.

Your solution still uses MLX arrays and its extension API. MLX schedules the
graph node, owns its buffers, and dispatches the Metal function, but your
solution owns the arithmetic inside that function. Your solution does not call
`mx.fast.rms_norm`,
`mx.fast.rope`, or an MLX-provided SiLU implementation.

## Why Fusion Helps

Week 1's Python `mlx.core` equations already become native GPU work inside the
lazy graph. Here, the useful question is how many operations, launches, and
memory passes that graph still describes.

For example, RMSNorm expressed as `mlx.core` operations casts, squares,
reduces, takes a reciprocal square root, multiplies, casts again, and applies a
learned weight. A compiler may fuse some adjacent element-by-element work, but
the row reduction is a boundary. Intermediate values and multiple dispatches
remain possible.

A single fused Metal kernel gives you explicit control over the whole operator:

- one dispatch replaces several graph operations;
- values stay in registers or SIMD-group storage between steps;
- float accumulation is used where numerical stability needs it;
- inputs are read once when practical, and only the final tensor is written;
- the grid matches decode shapes instead of a generic tensor operation.

So compare one purpose-built kernel with a graph of several general-purpose
kernels. The source language is not the point; the resulting work is.

## Task 1: RMSNorm

Start by replacing the fail-closed RMSNorm bodies: `tiny_llm_ext::rms_norm`,
`Week2RMSNorm::eval_cpu`, and
`Week2RMSNorm::eval_gpu` in `src/extensions/src/week2_kernels.cpp`, the `week2_rms_norm` function in
`src/extensions/src/week2_kernels.metal`, and `FastRMSNorm.__call__` in
`src/tiny_llm/week2_kernels.py`. The starter already provides the header,
binding, C++/Metal files, and CMake registration, so keep that API rather than
adding a parallel one.

Begin with one SIMD group per input row, then benchmark it. A 2,560-element hidden
row gives 32 lanes roughly 80 serial elements each; the optimized kernel launches 256
threads, or eight SIMD groups, per row. Each group reduces its portion with
`simd_sum`; lane zero writes eight partial sums to threadgroup memory; the first
SIMD group performs the second reduction:

```plain
sum_sq = simd_sum(each lane's partial sum)
inverse_rms = rsqrt(sum_sq / hidden_size + epsilon)
output[i] = input[i] * inverse_rms * weight[i]
```

For widths through 4096, each thread retains up to four input values through
the reduction and writes the result without rereading the row. Wider rows
retain the fixed-width fallback, and `FastRMSNorm.dispatch_counts` records
`register_cached` or `fixed_width_fallback`. All 256 lanes then normalize and
scale their strided elements. This fuses the
reduction and output pass into one dispatch and avoids materializing the
squared tensor. Instantiate the required kernel for bfloat16. Keep
the reduction, normalization, and weight multiplication in float, then cast the
final result once. The Python reference equation rounds once before applying the
weight, so compare the two with a tolerance rather than expecting bit-identical
results.

The C++ primitive validates shape and dtype, allocates the output through MLX,
binds the buffers and scalar constants, allocates eight float partial sums, and
launches one 256-thread group per row. Compare this two-level reduction with a
single-SIMD-group control to determine whether the extra parallelism offsets
the threadgroup reduction on the target machine.

Wire `FastRMSNorm` into every Week 2 norm as soon as the kernel works. Then run
the focused test and record the cumulative model result before touching RoPE:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k rms
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint rmsnorm --model qwen3-0.6b --num-seqs 1 \
  --min-input-len 128 --max-input-len 128 \
  --min-output-len 16 --max-output-len 16 --warmup 2 --prefill-logits last
```

## Task 2: RoPE

Next replace `tiny_llm_ext::rope`, `Week2RoPE::eval_cpu`, and
`Week2RoPE::eval_gpu` in `src/extensions/src/week2_kernels.cpp`, the
`week2_rope` function in `src/extensions/src/week2_kernels.metal`, and
`FastRoPE.__call__` in `src/tiny_llm/week2_kernels.py`.

Implement RoPE for the model's native `B, L, H, D` layout. A naive element
kernel calculates the same angle, sine, and cosine separately for both members
of every pair and again for every head. Instead, assign one thread a pair index
and a block of four heads. Compute the angle once, then rotate both elements of
that pair across the four heads:

```plain
angle = (batch_offset + token_position) * base ** (-pair / (dims / 2))
real' = real * cos(angle) - imag * sin(angle)
imag' = imag * cos(angle) + real * sin(angle)
```

Accept either one scalar offset or one offset per batch row in the Python
wrapper. Normalize both cases to an int32 array before dispatch. Supporting
per-batch offsets matters once requests at different decode positions share a
batch.

Unlike a graph that builds position arrays, gathers sine and cosine values,
splits the head, performs several element-by-element operations, and
concatenates the result, this kernel reads each input pair and writes each
rotated element directly. Reusing trigonometry across four heads is the key
optimization. Use Metal's `fast::exp2`, `fast::sin`, and `fast::cos` for the
BF16 path. Normalize a batch's offsets once in the model call,
outside the layer loop, instead of rebuilding the same array in every layer.

Replace the Python `mlx.core` RoPE in the model you have already optimized.
Test and measure that cumulative checkpoint before moving to SwiGLU:

```bash
pdm run test --week 2 --day 4 -- -k rope
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint rope --model qwen3-0.6b --num-seqs 1 \
  --min-input-len 128 --max-input-len 128 \
  --min-output-len 16 --max-output-len 16 --warmup 2 --prefill-logits last
```

## Task 3: SwiGLU

Finish the operator sequence with `tiny_llm_ext::swiglu`,
`Week2SwiGLU::eval_cpu`, and `Week2SwiGLU::eval_gpu` in
`src/extensions/src/week2_kernels.cpp`, the `week2_swiglu` function in
`src/extensions/src/week2_kernels.metal`, and `swiglu` in
`src/tiny_llm/week2_kernels.py`.

SwiGLU combines the gate and up branches:

```plain
output = (gate / (1 + exp(-gate))) * up
```

Implement it as one thread per element. That thread loads `gate` and `up`,
evaluates SiLU with one exponential, multiplies the branches, and performs one
output write. The Week 1 form is easier to inspect, but it describes `abs`,
`exp`, division, selection, and multiplication as separate array operations.
The fused kernel removes those intermediate tensors and dispatch boundaries.

Wire the fused expression into the model, then record the third checkpoint:

```bash
pdm run test --week 2 --day 4 -- -k swiglu
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-0.6b --num-seqs 1 \
  --min-input-len 128 --max-input-len 128 \
  --min-output-len 16 --max-output-len 16 --warmup 2 --prefill-logits last
```

## Task 4: Verify the Cumulative Model

Now verify the cumulative switches in `Qwen3ModelWeek2.__init__` and the call
sites in `Qwen3MultiHeadAttention.__call__` and `Qwen3MLP.__call__`. Task 4 is
composition work: it uses the three functions from Tasks 1-3 and adds no new
extension function.

Once all three kernels are exposed through C++ MLX primitives, run the complete
test file. Keep `qwen3_week1.py` on its Week 1 Python operators, and leave the
Week 2 interfaces reusable by the Week 3 serving model.

```bash
pdm run build-ext
pdm run test --week 2 --day 4
```

Use tolerance-based comparisons with the Python reference equations rather
than bit-for-bit equality. Cover both scalar and per-batch RoPE offsets. When
timing these lazy operations, call `mx.eval` inside every measured iteration.

The operator benchmark must also compare the same logical RoPE layout. Your
RoPE kernel accepts the model-native `B, L, H, D` tensor. `mx.fast.rope`
expects `B, H, L, D`, so transpose into that layout before the MLX call and
transpose its result back afterward. Without those transposes, a one-token
benchmark accidentally treats the head axis as sequence positions and the
timing no longer measures an equivalent operation.

## Benchmark Analysis: Decide Whether the Fused Kernels Stay

Measure the three cumulative checkpoints separately so their combined result
cannot hide a regression:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-simd-matmul \
  --variant week2-rmsnorm --variant week2-rope --variant week2-swiglu \
  --variant mlx --model qwen3-4b \
  --input-len 128 --output-len 129 --warmup 2 --prefill-logits last

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case quantized-matvec:decode:128 --case swiglu:decode:128 \
  --case swiglu:prefill:128 --warmup 4 --iterations 12 \
  --json-output week2-day4-attribution.json
```

Record one cumulative result per operator, then use the attribution run to
choose the next bottleneck. The complete campaign and reference attribution
are in the
[performance appendix](./appendix-performance.md).

In the older checked M4 Pro example, the fused kernels reduced the attributed
normalization/position/activation category by 79.0%. Re-profiling then placed
projections at 81.4% of decode attribution and 99.1% of 128-token prefill
attribution. The current next chapter is [tiled dense prefill attention](./week2-05-tiled-prefill-attention.md).
The old attribution remains a bounded example; it does not select a new
operator for this successor.

If you want to continue without writing one of these kernels, keep its public
course interface and delegate only that operator to the corresponding MLX
implementation. This local substitution still exercises the cached Week 2
model and the other course-owned operators; `--solution mlx` does not.

{{#include copyright.md}}
