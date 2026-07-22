# Week 2 Days 4-5: FlashAttention-2

In this chapter, we will implement a small FlashAttention-style Metal kernel
for the Week 2 Qwen3 serving pipeline. The goal is to learn the tiled,
IO-aware algorithm and map both matrix multiplications to Metal's public
`simdgroup_matrix` API. We will not reuse an MLX attention-kernel
implementation.

**📚 Readings**

- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [MLX Extension Development Guide](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Feature Set Tables](https://developer.apple.com/metal/feature-sets/)

## Why FlashAttention?

Standard attention computes:

1. `S = QK^T`
2. `P = softmax(S + mask)`
3. `O = PV`

This path materializes an `L x S` score tensor, and sometimes a separate
probability tensor, in device memory. FlashAttention instead streams K/V tiles
through on-chip memory and combines them with online softmax. Its auxiliary
memory is linear rather than quadratic in sequence length.

For Qwen3-4B, one BF16 score tensor at `L = S = 4096` occupies:

```plain
1 batch × 32 query heads × 4096 × 4096 × 2 bytes = 1 GiB
```

Avoiding that allocation is useful on unified-memory Apple silicon too.
However, lower memory use does not automatically mean lower latency. Two
optimized matrix multiplications can beat a fused kernel whose inner matrix
multiplications are poorly mapped to the GPU.

## Online Softmax Recap

For one query row, split keys and values into tiles `j = 1..T`:

$$
m^{(j)} = \max\left(m^{(j-1)}, \max(s^{(j)})\right)
$$

$$
l^{(j)} = e^{m^{(j-1)} - m^{(j)}} l^{(j-1)} + \sum e^{s^{(j)} - m^{(j)}}
$$

$$
o^{(j)} = e^{m^{(j-1)} - m^{(j)}} o^{(j-1)} + \sum e^{s^{(j)} - m^{(j)}} v^{(j)}
$$

At the end:

$$
o = \frac{o^{(T)}}{l^{(T)}}
$$

The running maximum prevents overflow. The correction factor
`exp(old_max - new_max)` rescales both the previous denominator and the
previous output when a later tile contains a larger score.

## The Qwen3 Target

The optimized course path is deliberately specialized instead of pretending
one kernel is ideal for every model:

```plain
Qwen3-4B attention
query heads: 32
KV heads:     8
head_dim:     128
dtype:        bfloat16
```

The supported dense Qwen3 models used by this course have `head_dim = 128`,
including Qwen3-4B. Do not derive this value as `hidden_size / num_heads`:
Qwen3 stores `head_dim` explicitly and its attention projection width can
differ from `hidden_size`.

## Task 1: Implement the Python Wrapper

```plain
src/tiny_llm/attention.py
```

Implement `flash_attention(query, key, value, scale=None, mask=None)` using the
same model-facing layout as the earlier attention chapter:

```plain
query: B..., H_q, L, E
key:   B..., H,   S, E
value: B..., H,   S, E
out:   B..., H_q, L, E
```

Flatten batch and head dimensions before calling C++, then restore the original
layout. Make Q, K, and V contiguous, but preserve their BF16 dtype.

Pass a small integer mask mode to the extension:

| Mode | Meaning | Mask buffer |
| ---: | --- | --- |
| 0 | no mask | one-element placeholder |
| 1 | causal | one-element placeholder |
| 2 | additive mask | contiguous broadcasted `(N, L, S)` FP32 array |

Do not construct a dense all-zero or causal mask. Doing so reintroduces the
quadratic allocation that the fused kernel is intended to avoid.

## Task 2: Implement the CPU Reference

```plain
src/extensions/src/tiny_llm_ext.h
src/extensions/bindings.cpp
src/extensions/src/flash_attention.cpp
src/extensions/CMakeLists.txt
```

Add the MLX primitive, binding, and CPU evaluator. The readable CPU path uses
FP32 and tiled online softmax with `Br = 32` and `Bc = 32`. It is a correctness
reference, not the performance path.

Map grouped-query heads with:

```plain
q_kv_ratio = num_heads / num_kv_heads
kv_head = query_head / q_kv_ratio
```

For causal attention, a key is visible when:

```plain
key_index <= query_index + (S - L)
```

The `S - L` offset is required when Q contains only the new tokens but K/V also
contain cached tokens. Skip a K/V tile entirely when all of its keys are in the
future.

## Task 3: Implement the Metal Kernels

```plain
src/extensions/src/flash_attention.metal
src/extensions/src/flash_attention.cpp
src/extensions/CMakeLists.txt
```

### Why the First Scalar Mapping Is Slow

A kernel can implement the FlashAttention recurrence correctly and still be
slower than ordinary attention. The original mapping had four main problems:

- It launched `32 × 32 = 1024` threads per threadgroup. A hardware maximum is
  a budget, not an occupancy target.
- Each score used a serial 128-element dot product instead of a matrix
  instruction.
- `P @ V` looped over 128 output columns and performed a SIMD reduction for
  every column.
- Q/K/V were converted to FP32 and dense no-mask/causal arrays were created,
  even though Week 2 Qwen activations are BF16.

The ordinary implementation writes a quadratic score tensor, but its QK and
PV operations are highly optimized matrix kernels. Eliminating the tensor
cannot compensate for slow scalar arithmetic.

### Keep a General Fallback

Use a small scalar kernel for FP32 head dimensions below 128. A useful mapping
is a `16 × 32` query/key tile with 512 threads: one SIMD group owns one query
row, and one lane owns one key. This path keeps the implementation general and
testable.

The performance path is a separate BF16, `E = 128` specialization. Specialize
the common model shape instead of adding branches and dynamic loops to its hot
path.

### BF16 SIMD-Matrix Tile

Use this tile on the optimized path:

```plain
BQ = 64 query rows
BK = 32 key/value rows
D  = 128
8 SIMD groups × 32 lanes = 256 threads
```

Each SIMD group owns eight query rows. Metal's public
`simdgroup_matrix<T, 8, 8>` distributes one 8×8 fragment across its 32 lanes,
with two fragment elements per lane.

```plain
one SIMD group

Q fragment [8, 8] × Kᵀ fragment [8, 8] -> score fragment [8, 8]

D=128: 16 multiply-accumulate steps
BK=32: 4 score fragments

P fragment [8, 8] × V fragment [8, 8] -> output fragment [8, 8]

BK=32: 4 multiply-accumulate steps
D=128: 16 output fragments
```

Use native `bfloat` matrix operands and FP32 accumulator fragments. Keep the
row maxima, row sums, and online-softmax output accumulators in FP32, then
convert only the final output to BF16. This matches the Week 2 model without
whole-tensor dtype conversions.

### Threadgroup Memory and Registers

Stage Q once and stream K/V through a reused allocation:

```plain
threadgroup BF16 Q tile:  64 × 130
threadgroup BF16 K/V:    128 × 34
total:                   about 25 KiB
```

K is transposed while loading so QK can consume ordinary row-major 8×8
fragments. After QK and softmax, reuse the same allocation for row-major V.
The output's 16 matrix fragments and online-softmax statistics remain in
registers.

The two-element padding makes each BF16 row stride odd when measured in
32-bit threadgroup-memory banks. Padding should be chosen from the element
width and bank mapping; copying a padding value from an FP32 or CUDA kernel is
not generally correct.

### Per-Tile Sequence

For every K/V tile:

1. Cooperatively load and transpose K into threadgroup memory.
2. Compute four 8×8 score fragments with
   `simdgroup_multiply_accumulate` over the 16 D fragments.
3. Apply scale and mask, then reduce each eight-row fragment's maximum.
4. Update online-softmax maxima and sums in FP32.
5. Convert the probabilities to BF16 matrix fragments.
6. Reuse the K allocation for V and compute all 16 output fragments.
7. Rescale the previous output whenever the running maximum changes.

For causal prefill, stop the K/V loop at the last tile that can contain a
visible key. This avoids approximately half the matrix work when `L = S`.

### Do Not Import an MLX Kernel

The extension necessarily uses MLX's public C++ extension and command-encoder
interfaces, but the Metal shader should include only public Metal headers:

```metal
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
```

Do not include MLX Steel headers or instantiate an MLX attention template. The
point of this task is to implement and understand the fragment mapping,
softmax recurrence, storage layout, and synchronization directly.

### Correctness Cases

Test more than aligned self-attention:

- no mask, additive mask, and causal mask;
- BF16 D=128 and the FP32 fallback;
- partial query and key tiles, such as `L=35, S=47`;
- grouped-query attention;
- `L != S` causal offsets.

Evaluate the lazy result before comparing it. A kernel can compile and appear
to run while a lane-coordinate or partial-tile error remains hidden.

### Benchmark the GPU Work, Not Graph Construction

MLX evaluates lazily. Initialize Q/K/V before timing and force every iteration
to complete:

```python
def evaluate_attention(fn):
    result = fn()
    mx.eval(result)
    mx.synchronize()
    return result
```

On an Apple M4 Pro with MLX 0.29.1, single-request Qwen3-4B causal prefill
(`Hq=32`, `Hkv=8`, `D=128`, BF16) measured approximately:

| Tokens | Explicit attention | MLX fused | Course SIMD-matrix kernel |
| ---: | ---: | ---: | ---: |
| 128 | 0.293 ms | 0.205 ms | 0.538 ms |
| 512 | 1.272 ms | 0.569 ms | 1.899 ms |
| 1024 | 4.755 ms | 1.635 ms | 5.418 ms |
| 2048 | 17.084 ms | 5.584 ms | 19.015 ms |
| 4096 | 66.724 ms | 21.181 ms | 72.717 ms |

The standalone educational kernel does not beat the explicit implementation
on this GPU. At 4096 tokens it is about 9% slower, but it avoids the explicit
path's roughly 1 GiB score tensor. That is still a meaningful implementation:
it demonstrates the correct IO-aware algorithm and has bounded scratch memory.

It also shows why production FlashAttention kernels are complex. MLX's fused
implementation remains about 3.4× faster at 2048 tokens because it adds deeper
specialization and scheduling work beyond merely using matrix instructions.
Do not report memory efficiency as a latency speedup.

### Does FlashAttention Make Sense on Metal?

Yes for memory-bounded prefill, but not for exactly the same reasons or with
the same implementation as an NVIDIA kernel.

Apple GPUs have unified memory, SIMD width 32, Metal threadgroup memory, and a
public 8×8 SIMD-group matrix API. They do not expose the same warp-level tensor
core and asynchronous-copy model as H100/H200. Tile size, bank padding,
occupancy, and synchronization must therefore be retuned for Metal.

Apple's feature tables list SIMD-scoped matrix multiply beginning with Apple
GPU family 7, which includes M1-series GPUs. Check the runtime GPU family if the
course is extended beyond Apple silicon rather than assuming this path exists
on every Metal device.

FlashAttention is less compelling for one-token decode. Decode has almost no
query-tile reuse and is closer to a matrix-vector problem; use a dedicated
decode or paged-attention kernel for that phase.

## Task 4: Model Integration

```plain
src/tiny_llm/qwen3_week2.py
```

Connect the kernel to `Qwen3MultiHeadAttention` and propagate
`enable_flash_attn` through the model. Preserve Q, K, V, and the result as BF16;
do not cast the full tensors to FP32. Pass causal mode directly so the wrapper
does not allocate an `L × S` mask.

Run generation with FlashAttention enabled:

```bash
pdm run main --solution tiny_llm --loader week2 --model qwen3-4b --enable-flash-attn
```

Benchmark both paths:

```bash
pdm run bench --solution tiny_llm --loader week2 --model qwen3-4b
pdm run bench --solution tiny_llm --loader week2 --model qwen3-4b --enable-flash-attn
```

{{#include copyright.md}}
