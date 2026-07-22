# 🚧 Week 2 Day 5: Fast Operators

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

Implement one SIMD group per input row. Each lane accumulates a regularly spaced subset
of squared values in `float`, and `simd_sum` combines the 32 partial sums:

```plain
sum_sq = simd_sum(each lane's partial sum)
inverse_rms = rsqrt(sum_sq / hidden_size + epsilon)
output[i] = input[i] * inverse_rms * weight[i]
```

The same lanes then normalize and scale their elements. This fuses the
reduction and output pass into one dispatch and avoids materializing the
squared tensor. Instantiate the kernel for float32, float16, and bfloat16.
For half-precision inputs, round the normalized value to the model dtype before
applying the learned weight. That matches the readable operator's numerical
boundary and prevents tiny per-layer differences from compounding in deeper
models.

The C++ primitive validates shape and dtype, allocates the output through MLX,
binds the buffers and scalar constants, and launches exactly one 32-lane SIMD
group per row.

## Task 2: RoPE

Implement RoPE for the model's native `B, L, H, D` layout. Give each Metal
thread one output element. From its flat index, recover batch, token, head, and
head-dimension coordinates; then compute the pair index and rotation angle:

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
splits the head, performs several element-by-element operations, and concatenates the
result, this kernel reads each input pair and writes each rotated element
directly. It also avoids layout transposes by accepting the model's existing
layout.

## Task 3: SwiGLU

SwiGLU combines the gate and up branches:

```plain
output = (gate / (1 + exp(-gate))) * up
```

Implement it as one thread per element. That thread loads `gate` and `up`,
evaluates SiLU, multiplies the branches, and performs one output write. The
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
pdm run test --week 2 --day 5
```

Compare against the readable equations with tolerances rather than bit-for-bit
equality. Test RoPE with scalar and per-batch offsets. Always call `mx.eval`
inside a timed iteration when measuring these lazy operations.

## Expected Performance Contribution

**Estimated decode improvement: 5-15% after quantized matmul.** Each operator
is small, but a Qwen3 decode token invokes it many times across all layers, so
eliminating launches and intermediate memory traffic accumulates. The range is
an end-to-end estimate and overlaps with later changes; it is not additive.

{{#include copyright.md}}
