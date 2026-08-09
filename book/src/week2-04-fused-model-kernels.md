# 🚧 Week 2 Day 4: Fused Model Kernels

> **Status: Experimental.** See the
> [Week 2 verification matrix](./week2-overview.md#verification-status) for
> what is continuously tested, locally measured, and still under review.

Day 3 removed the largest projection gap. Day 4 now targets RMSNorm, RoPE, and
SwiGLU, which recur around those projections in every transformer layer. Week 1
expresses them as Python `mlx.core` equations. Week 2 keeps those implementations
intact and asks you to write three Metal kernels behind a separate interface:

```plain
src/tiny_llm/week2_kernels.py
src/extensions/src/week2_kernels.cpp
src/extensions/src/week2_kernels.metal
```

Your solution still uses MLX arrays and its extension API. MLX schedules the
graph node, owns its buffers, and dispatches the Metal function, but your
solution owns the arithmetic inside that function. Your solution does not call
`mx.fast.rms_norm`,
`mx.fast.rope`, or an MLX-provided SiLU implementation.

> **Optional profiling evidence.** The Day 3 kernel-group replay and the
> [reference-solution attribution](./appendix-performance.md#checked-operator-attribution-that-selects-each-chapter)
> show the pointwise cluster behind the optimized projections. They explain the
> chapter order but are not prerequisites or acceptance gates.

## Why Fusion Helps

Week 1's Python `mlx.core` equations already run as native GPU kernels inside
the lazy graph. The important difference is how many operations and memory
passes the graph describes.

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

The useful comparison is not "Metal versus Python arithmetic," but one
purpose-built kernel versus a graph of several general-purpose kernels.

## Task 1: RMSNorm

Modify `tiny_llm_ext::rms_norm`, `Week2RMSNorm::eval_cpu`, and
`Week2RMSNorm::eval_gpu` in `src/extensions/src/week2_kernels.cpp`, the
`week2_rms_norm` function in `src/extensions/src/week2_kernels.metal`, and
`FastRMSNorm.__call__` in `src/tiny_llm/week2_kernels.py`. The starter header,
binding, C++/Metal files, and CMake registration already exist for this
checkpoint; replace the fail-closed bodies instead of adding parallel APIs.

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

All 256 lanes then normalize and scale their strided elements. This fuses the
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

Integrate `FastRMSNorm` into every Week 2 norm immediately, run the RMSNorm
tests, and record the cumulative model result before writing RoPE:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k rms
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint rmsnorm --model qwen3-4b
```

## Task 2: RoPE

Modify `tiny_llm_ext::rope`, `Week2RoPE::eval_cpu`, and
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

Replace the Python `mlx.core` RoPE in the already optimized model, then test and measure
that cumulative checkpoint before implementing SwiGLU:

```bash
pdm run test --week 2 --day 4 -- -k rope
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint rope --model qwen3-4b
```

## Task 3: SwiGLU

Modify `tiny_llm_ext::swiglu`, `Week2SwiGLU::eval_cpu`, and
`Week2SwiGLU::eval_gpu` in `src/extensions/src/week2_kernels.cpp`, the
`week2_swiglu` function in `src/extensions/src/week2_kernels.metal`, and
`swiglu` in `src/tiny_llm/week2_kernels.py`.

SwiGLU combines the gate and up branches:

```plain
output = (gate / (1 + exp(-gate))) * up
```

Implement it as one thread per element. That thread loads `gate` and `up`,
evaluates SiLU with one exponential, multiplies the branches, and performs one
output write. The Week 1 form is easier to inspect, but it describes `abs`,
`exp`, division, selection, and multiplication as separate array operations.
The fused kernel removes those intermediate tensors and dispatch boundaries.

Integrate the fused expression immediately and record the third checkpoint:

```bash
pdm run test --week 2 --day 4 -- -k swiglu
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-4b
```

## Task 4: Verify the Cumulative Model

Verify the cumulative switches in `Qwen3ModelWeek2.__init__` and the call sites
in `Qwen3MultiHeadAttention.__call__` and `Qwen3MLP.__call__`. Task 4 should not
introduce another extension function; it composes the three functions from
Tasks 1-3.

After exposing all three kernels through C++ MLX primitives, run the complete
test file to verify their composition. Keep `qwen3_week1.py` on its Week 1
Python operators, and make the Week 2 interfaces reusable by the Week 3 serving model.

```bash
pdm run build-ext
pdm run test --week 2 --day 4
```

Compare against the Python reference equations with tolerances rather than bit-for-bit
equality. Test RoPE with scalar and per-batch offsets. Always call `mx.eval`
inside a timed iteration when measuring these lazy operations.

The operator benchmark must also compare the same logical RoPE layout. Your
RoPE kernel accepts the model-native `B, L, H, D` tensor. `mx.fast.rope`
expects `B, H, L, D`, so transpose into that layout before the MLX call and
transpose its result back afterward. Without those transposes, a one-token
benchmark accidentally treats the head axis as sequence positions and the
timing no longer measures an equivalent operation.

## Benchmark Analysis: Decide Whether the Fused Kernels Stay

Keep the three cumulative checkpoints separate so a regression cannot hide
inside their combined gain:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-quantized-matvec \
  --variant week2-rmsnorm --variant week2-rope --variant week2-swiglu \
  --variant mlx --model qwen3-4b \
  --input-len 128 --output-len 129 --warmup 2 --prefill-logits last

pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section model-kernels --context 128
```

Attach each cumulative model row, the three Python-reference/optimized/MLX operator
rows, and the direct dispatch trace. Let the matched benchmark results decide
whether the kernels stay.

Continue to Day 5 when all three correctness gates pass, the direct
fused-dispatch source trace reaches the intended kernels, the cumulative rows
retain the gain, and the three operator comparisons justify keeping the fused
implementations. Day 5 then tests whether attention is the next removable gap
by sweeping cached context and query length before setting a dispatch guard.

> **Optional profiling evidence.** The
> [reference checkpoint](./appendix-performance.md#day-4-fused-model-kernels)
> pairs the cumulative and operator measurements with an updated
> attribution. That attribution can explain the transition, but it does not
> replace the checkpoint evidence above.

{{#include copyright.md}}
