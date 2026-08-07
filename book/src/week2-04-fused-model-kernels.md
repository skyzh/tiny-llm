# 🚧 Week 2 Day 4: Fused Model Kernels

> **Status: Experimental.** See the
> [Week 2 verification matrix](./week2-overview.md#verification-status) for
> what is continuously tested, locally measured, and still under review.

The Day 3 profile should now show many smaller pointwise and reduction
dispatches behind the optimized projections.
RMSNorm, RoPE, and SwiGLU recur in every transformer layer, so their cumulative
GPU duration—not an imagined single slow call—makes them the next target. Week
1 expresses them as readable `mlx.core` equations. Confirm the cluster with the
Day 2 kernel-group replay; the
[reference-solution profile](./appendix-performance.md#the-kernel-profile-that-selects-each-chapter)
shows the expected transition. Week 2 keeps those implementations intact and
asks you to write three Metal kernels behind a separate interface:

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

## Why a Metal Kernel Helps

Calling the Week 1 code "Python" does not mean Python visits every tensor
element. Python builds a lazy graph whose individual array operations already
run as native kernels. The important difference is how many operations and
memory passes the graph describes.

For example, readable RMSNorm casts, squares, reduces, takes a reciprocal square
root, multiplies, casts again, and applies a learned weight. A compiler may fuse
some adjacent element-by-element work, but the row reduction is a boundary. Intermediate
values and multiple dispatches remain possible.

A Metal kernel in your solution gives you explicit control over the whole
operator:

- one dispatch replaces several graph operations;
- values stay in registers or SIMD-group storage between steps;
- float accumulation is used where numerical stability needs it;
- inputs are read once when practical, and only the final tensor is written;
- the grid matches decode shapes instead of a generic tensor operation.

That is the useful comparison: not "Metal versus Python arithmetic," but one
purpose-built kernel versus a graph of several general-purpose kernels.

## Task 1: RMSNorm

Begin with one SIMD group per input row, then profile it. A 2,560-element hidden
row gives 32 lanes too much serial work. The optimized kernel launches 256
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
final result once. The readable Week 1 equation rounds once before applying the
weight, so compare the two with a tolerance rather than expecting bit-identical
results. The single final cast also tracks the MLX model more closely in the
Qwen3-4B end-to-end correctness test.

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

Replace the readable RoPE in the already optimized model, then test and measure
that cumulative checkpoint before implementing SwiGLU:

```bash
pdm run test --week 2 --day 4 -- -k rope
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint rope --model qwen3-4b
```

## Task 3: SwiGLU

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

After exposing all three kernels through C++ MLX primitives, run the complete
test file to verify their composition. Keep `qwen3_week1.py` on its readable
operators, and make the Week 2 interfaces reusable by the Week 3 serving model.

```bash
pdm run build-ext
pdm run test --week 2 --day 4
```

Compare against the readable equations with tolerances rather than bit-for-bit
equality. Test RoPE with scalar and per-batch offsets. Always call `mx.eval`
inside a timed iteration when measuring these lazy operations.

The operator benchmark must also compare the same logical RoPE layout. Your
RoPE kernel accepts the model-native `B, L, H, D` tensor. `mx.fast.rope`
expects `B, H, L, D`, so transpose into that layout before the MLX call and
transpose its result back afterward. Without those transposes, a one-token
benchmark accidentally treats the head axis as sequence positions and the
timing no longer measures an equivalent operation.

## Benchmark Analysis: Select Day 5

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

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case swiglu:decode:32 --case swiglu:decode:128 \
  --case swiglu:decode:160 --warmup 4 --iterations 12
```

Use a source-enabled pointwise capture for your implementation only as part of
the optional profiling appendix:

```bash
CMAKE_ARGS="-DMLX_METAL_DEBUG=ON" pdm run build-ext
MLX_METAL_DEBUG=1 MTL_CAPTURE_ENABLED=1 pdm run capture-week2-shader \
  --solution tiny_llm --workload pointwise --iterations 10 \
  --output /tmp/week2-day4-pointwise.gputrace
```

Attach each cumulative model row, the three readable/optimized/MLX operator
rows, and the post-SwiGLU kernel-group profile. Use Xcode to verify the three
pipeline identities, but let your benchmark results decide whether the kernels
stay.

After the pointwise cluster shrinks, sweep cached context rather than assuming
attention is next. Continue to Day 5 when attention is a removable gap at
`S <= 128` and the projection operators are already close to their
denominator. Measurements at `S >= 160` define the fallback range; they do not
justify a kernel that dispatches only through 128. Day 5 must still prove that
its replacement moves a complete-model workload that actually enters that
guard. The
[reference checkpoint](./appendix-performance.md#day-4-fused-model-kernels)
pairs the cumulative gains with all three operator microbenchmarks and the
updated attribution. Use the optional trace to verify that the intended three
fused kernels ran; the new context sweep, not the absolute projection share,
selects attention next.

{{#include copyright.md}}
