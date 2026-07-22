# 🚧 Week 2 Day 4: Fast Kernels

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
squared tensor. Instantiate the kernel for float32, float16, and bfloat16.
For half-precision inputs, round the normalized value to the model dtype before
applying the learned weight. That matches the readable operator's numerical
boundary and prevents tiny per-layer differences from compounding in deeper
models.

The C++ primitive validates shape and dtype, allocates the output through MLX,
binds the buffers and scalar constants, allocates eight float partial sums, and
launches one 256-thread group per row. The two-level reduction was the largest
small-operator improvement in the measured stack; a single SIMD group left too
little parallel work available.

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

## Task 3: SwiGLU

SwiGLU combines the gate and up branches:

```plain
output = (gate / (1 + exp(-gate))) * up
```

Implement it as one thread per element. That thread loads `gate` and `up`,
evaluates SiLU with one exponential, multiplies the branches, and performs one output write. The
Week 1 form is easier to inspect, but it describes `abs`, `exp`, division,
selection, and multiplication as separate array operations. The fused kernel
removes those intermediate tensors and dispatch boundaries.

## Task 4: Integrate and Test

Expose all three kernels through a C++ MLX primitive and thin Python wrappers.
Import those wrappers only in `qwen3_week2.py`; `qwen3_week1.py` must continue
using its readable operators. Week 3 should import the Week 2 interfaces so the
serving model does not regress.

```bash
pdm run build-ext
pdm run test --week 2 --day 4
```

Compare against the readable equations with tolerances rather than bit-for-bit
equality. Test RoPE with scalar and per-batch offsets. Always call `mx.eval`
inside a timed iteration when measuring these lazy operations.

## Expected Performance Contribution

**Estimated decode improvement: 5-15% after quantized matmul.** Each operator
is small, but a Qwen3 decode token invokes it many times across all layers, so
eliminating launches and intermediate memory traffic accumulates. In current
isolated measurements, course RMSNorm is within about 2-9% of MLX's operator,
RoPE within about 7-14%, and SwiGLU approximately equal or slightly faster.
The earlier one-SIMD-group RMSNorm materially limited the full model; its
256-thread two-level reduction closed most of that gap. These isolated ratios
are more stable than attributing an end-to-end percentage to three overlapping
replacements, and the 5-15% range is not additive with other chapters.

{{#include copyright.md}}
