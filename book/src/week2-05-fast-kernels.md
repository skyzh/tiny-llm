# 🚧 Week 2 Day 5: Fast Kernels

> 🚧 This newly introduced chapter is a work in progress.

Week 1 expresses RMSNorm, RoPE, and SiLU as readable `mlx.core` equations.
Week 2 keeps those implementations intact and writes three course-owned Metal
kernels behind a separate interface:

```plain
src/tiny_llm/week2_kernels.py
src/extensions/src/week2_kernels.cpp
src/extensions/src/week2_kernels.metal
```

We still use MLX arrays and its extension API. MLX schedules the graph node,
owns its buffers, and dispatches the Metal function, but the arithmetic inside
that function is ours. The required solution does not call `mx.fast.rms_norm`,
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

A course-owned Metal kernel gives us explicit control over the whole model
operator:

- one dispatch replaces several graph operations;
- values stay in registers or SIMD-group storage between steps;
- float accumulation is used where numerical stability needs it;
- inputs are read once when practical, and only the final tensor is written;
- the grid matches decode shapes instead of a generic tensor operation.

That is the useful comparison: not "Metal versus Python arithmetic," but one
purpose-built kernel versus a graph of several general-purpose kernels.

## Task 1: RMSNorm

Begin with one SIMD group per input row, then profile it. A 1024-element hidden
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
squared tensor. Instantiate the kernel for float32, float16, and bfloat16. Keep
the reduction, normalization, and weight multiplication in float, then cast the
final result once. The readable Week 1 equation rounds once before applying the
weight, so compare the two with a tolerance rather than expecting bit-identical
results. The single final cast also tracks the MLX model more closely in the
Qwen3-4B end-to-end correctness test.

The C++ primitive validates shape and dtype, allocates the output through MLX,
binds the buffers and scalar constants, allocates eight float partial sums, and
launches one 256-thread group per row. The two-level reduction was the largest
small-operator improvement in the measured stack; a single SIMD group left too
little parallel work available.

Integrate `FastRMSNorm` into every Week 2 norm immediately, run the RMSNorm
tests, and record the cumulative model result before writing RoPE:

```bash
pdm run build-ext
pdm run test --week 2 --day 5 -- -k rms
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint rmsnorm --model qwen3-0.6b
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
optimization. For float16 and bfloat16, use Metal's `fast::exp2`, `fast::sin`,
and `fast::cos`; retain precise functions for float32 so strict correctness
tests remain meaningful. Normalize a batch's offsets once in the model call,
outside the layer loop, instead of rebuilding the same array in every layer.

Replace the readable RoPE in the already optimized model, then test and measure
that cumulative checkpoint before implementing SwiGLU:

```bash
pdm run test --week 2 --day 5 -- -k rope
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint rope --model qwen3-0.6b
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
pdm run test --week 2 --day 5 -- -k swiglu
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-0.6b
```

## Task 4: Verify the Cumulative Model

At this point all three kernels have already been exposed through C++ MLX
primitives, integrated, and measured. Run the complete test file now to verify
their composition. `qwen3_week1.py` must still use its readable operators, and
Week 3 should import the Week 2 interfaces so the serving model does not regress.

```bash
pdm run build-ext
pdm run test --week 2 --day 5
```

Compare against the readable equations with tolerances rather than bit-for-bit
equality. Test RoPE with scalar and per-batch offsets. Always call `mx.eval`
inside a timed iteration when measuring these lazy operations.

The operator benchmark must also compare the same logical RoPE layout. The
course kernel accepts the model-native `B, L, H, D` tensor. `mx.fast.rope`
expects `B, H, L, D`, so transpose into that layout before the MLX call and
transpose its result back afterward. Without those transposes, a one-token
benchmark accidentally treats the head axis as sequence positions and the
timing no longer measures an equivalent operation.

## Expected Performance Contribution

**Measured operator improvement: about 34-39% over the readable Week 1
equations at decode shapes.** In repeated Qwen3-0.6B M4 Pro runs, RMSNorm was
about 1.36x faster, RoPE about 1.34x, and SwiGLU about 1.39x. The cumulative
course ladder integrates them after decode attention: RMSNorm increased 143.42
to 193.18 tok/s (+34.7%), RoPE increased 193.18 to 222.51 (+15.2%), and SwiGLU
increased 222.51 to 242.67 (+9.1%). These sequential deltas already include all
earlier changes; do not add them to isolated or reverse-ablation percentages.

{{#include copyright.md}}
