# Week 2 Day 4: Fast Model Operators (WIP)

> This newly introduced chapter is a work in progress.

The Week 1 operators favor readable equations. That makes them useful for
learning, but it also makes one model operation appear to MLX as a chain of
smaller array operations. Week 2 replaces those chains with specialized or
fused work while preserving the same math.

```plain
src/tiny_llm/week2_kernels.py
src/tiny_llm_ref/week2_kernels.py
```

## Python Is Not Doing the Arithmetic

Calling the Week 1 implementation "Python" can be misleading. Python builds a
lazy MLX computation graph; MLX still executes its array operations with native
CPU or Metal kernels. There is no Python loop visiting every tensor element.

The difference is the amount of work described by each graph node. Consider
the readable RMSNorm:

```python
x = x.astype(mx.float32)
x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
x = x.astype(orig_dtype)
return x * weight.astype(orig_dtype)
```

This graph contains casts, a square, a reduction, an inverse square root, and
several multiplications. MLX can fuse some compatible per-element operations,
but a reduction is an important boundary. Values may need to be written to and
read from memory between stages, and each dispatched kernel has a fixed launch
cost.

`mx.fast.rms_norm`, by contrast, presents RMSNorm as one operation. Its backend
can reduce a row, normalize it, and apply the weight in a purpose-built kernel,
keeping partial results in fast on-chip storage where possible.

The optimization pattern is the same for the three operators in this chapter:

| Operator | Readable Week 1 graph | Faster Week 2 path | Main saving |
| --- | --- | --- | --- |
| RMSNorm | Cast, square, mean, reciprocal square root, scale | `mx.fast.rms_norm` | Fused reduction and scaling |
| RoPE | Gather sine/cosine tables, multiply, add, concatenate | `mx.fast.rope` | Fewer temporary tensors and dispatches |
| SwiGLU | Manual SiLU expression, then multiply by the up branch | One fusible MLX expression | One pass over the activation |

These operators are smaller than the model's linear layers, but decode invokes
them repeatedly with very little work in each invocation. In a 28-layer Qwen3
model, one generated token passes through 113 RMSNorm calls, 56 RoPE
applications, and 28 SwiGLU activations. Reducing a small fixed cost matters
when it is paid that many times for every token.

## Task 1: Fast RMSNorm

Implement `FastRMSNorm` with `mx.fast.rms_norm`. Preserve the constructor and
call shape of the readable `RMSNorm` so the model can select the implementation
without changing its surrounding logic.

The specialized kernel improves two things:

1. **Memory traffic.** The readable graph can materialize values around the
   row-wise reduction. A fused kernel can load an input row, accumulate its sum
   of squares, and reuse the data while normalizing and scaling.
2. **Dispatch overhead.** One model-level operation becomes one optimized
   primitive instead of several generic graph operations.

Keep the accumulation behavior in mind when testing. The fast kernel and the
readable float32 expression may round values in a different order, so compare
them with a tolerance rather than requiring bit-for-bit equality.

## Task 2: Fast RoPE

Implement `FastRoPE` with `mx.fast.rope`. The Qwen model stores tensors as
`B, L, H, D`, while the primitive expects `B, H, L, D`, so transpose on entry
and restore the original layout on return. Support both one scalar offset and
one offset per batch element.

The Week 1 implementation makes the rotation explicit. It gathers sine and
cosine values for the current positions, splits each head into two halves,
computes four products and two sums, then concatenates the result. Those steps
are easy to inspect, but they introduce several graph nodes and temporary
values.

The fast primitive applies the same rotation in a tuned kernel. MLX receives the
position offset directly and does not need the Python model to assemble the
rotation from separate array operations. This is particularly useful in decode,
where the model rotates one new query and key position per layer while the
offset advances on every token.

The transposes do not perform arithmetic. They describe the layout expected by
the kernel and restore the model-facing layout afterward. Whether a transpose
requires a copy depends on how the consuming kernel handles the resulting
strides, so keep layout changes next to the primitive and verify them in the
end-to-end benchmark.

## Task 3: SwiGLU

Express SwiGLU as one array expression:

```python
gate * mx.sigmoid(gate) * up
```

The Week 1 `silu` spells out a numerically stable sigmoid using `abs`, `exp`,
division, `where`, and multiplication. It is excellent for showing the formula,
but the MLP only needs the combined result `SiLU(gate) * up`.

Keeping the Week 2 form in one expression gives MLX a simple per-element graph
that it can fuse into one generated kernel. The kernel can load `gate` and `up`,
compute the activation, multiply the branches, and write the result once. This
optimization does not require a handwritten Metal extension; MLX generates the
fused backend work from the lazy graph.

Do not change the Week 1 `silu` implementation. The separate Week 2 function
makes the readable and optimized forms directly comparable and keeps the course
incremental.

## Why This Helps Decode More Than Prefill

During prefill, large matrix multiplications and attention tiles provide enough
work to amortize a kernel launch. During single-token decode, tensor shapes are
small in the token dimension, so fixed costs account for a larger fraction of
latency. Decode also repeats the same sequence of operators once per layer and
once per generated token.

The useful mental model is therefore not "Metal is faster than Python." Both
versions already use native MLX kernels. The improvement comes from:

- dispatching fewer kernels;
- reading and writing fewer intermediate tensors;
- using a kernel specialized for the reduction or rotation;
- exposing a simple graph that MLX can fuse; and
- paying those fixed costs fewer times across every layer and token.

These changes should be evaluated end to end. A microbenchmark can show that an
individual operator is cheaper, but token throughput is the acceptance metric.
Synchronize lazy execution with `mx.eval`, warm up both paths, stop other
CPU- and GPU-intensive workloads, and compare several runs under the same
thermal and power conditions.

## Task 4: Integrate the Layer

Update `qwen3_week2.py` to import the Week 2 operations. `qwen3_week3.py` must
reuse the same interfaces so serving features do not regress to the readable
Week 1 kernels.

```bash
pdm run test --week 2 --day 4
```

The tests compare the fast operations with the readable versions, exercise
scalar and per-batch RoPE offsets, and verify the Week 1/2/3 boundaries.

{{#include copyright.md}}
