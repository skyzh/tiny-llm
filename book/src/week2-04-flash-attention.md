# Week 2 Day 4-5: Flash Attention 2

In this chapter, we will implement Flash Attention 2 for the Week 2 Qwen2 serving pipeline. The goal is to replace the regular attention path with a tiled implementation to reduce memory bandwidth and increase throughput, especially for long contexts. 

**📚 Readings**

- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [MLX Extension Development Guide](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [MLX steel attention kernel (reference)](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h)

## Why Flash Attention?

The key idea from the FlashAttention papers is that attention is often **IO-bound**, not FLOP-bound.

In the standard implementation, we compute:

1. `S = QK^T`
2. `P = softmax(S + mask)`
3. `O = PV`

This path materializes large `L x S` tensors (`S` and often `P`) in global memory. For long contexts, repeatedly writing and reading these tensors dominates runtime.

For example, if `L = S = 4096`:

```plain
One L x S matrix: 4096 x 4096 = 16,777,216 elements
float32 storage: ~64 MB per matrix per head
Scores + probabilities: ~128 MB temporary memory per head
```

So even before counting Q/K/V and output tensors, memory traffic is already huge.

### IO-Aware Exact Attention

FlashAttention avoids this bottleneck by tiling Q/K/V into on-chip memory (cache / shared memory), and combining each tile with **online softmax** updates. Instead of storing the full attention matrix, it keeps only per-row running statistics (`m`, `l`) and partial output (`o`).

This gives three practical benefits:

- **Exactness**: same result as standard softmax attention (not an approximation).
- **Lower memory**: activation memory scales linearly with sequence length instead of quadratically.
- **Higher throughput**: fewer high-bandwidth-memory accesses, which is usually the real bottleneck.

## Online Softmax Recap

For one query row, split keys/values into tiles `j = 1..T`:

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

This is the core numerical trick used by both the CPU and GPU kernels in this chapter, and the rest of the implementation is mostly about mapping this update rule to CPU loops and Metal threadgroups.

## Task 1: Implement `flash_attention` Wrapper

```
src/tiny_llm/attention.py
```

Implement `flash_attention(query, key, value, scale=None, mask=None)` so it matches the extension API in `tiny_llm_ext`.

Follow the same shape convention as Week 1 and Week 2 attention:

```plain
query: B..., H_q, L, E
key:   B..., H,   S, E
value: B..., H,   S, E
mask:  B..., H_q, L, S
out:   B..., H_q, L, E
```

The wrapper should compute `factor` using `mx.rsqrt` when `scale` is `None`, flatten batch and head dimensions before calling into C++, and reshape the output back to the original layout. Make sure `query`, `key`, and `value` are contiguous before calling the extension. For `mask`, always broadcast to `B..., H_q, L, S`, reshape to `(N, L, S)`, and cast to `float32` so that CPU and GPU kernels receive exactly the same dtype.

## Task 2: Implement `flash_attention` (CPU version)

```
src/extensions/src/tiny_llm_ext.h
src/extensions/bindings.cpp
src/extensions/src/flash_attention.cpp
src/extensions/CMakeLists.txt
```

In this task, add the new MLX primitive and its CPU implementation. The structure is the same as the quantized matmul chapter: declare the primitive in `tiny_llm_ext.h`, expose it in `bindings.cpp`, and register `flash_attention.cpp` in `CMakeLists.txt`.

Before creating the lazy output array, validate all shape and dtype constraints in C++: inputs should be 3D float32 tensors, `num_heads` must be divisible by `num_kv_heads`, and head mapping between Q and KV batches must be consistent.

Then implement `FlashAttention::eval_cpu(...)` with tiled online softmax. Use `Br = 32` and `Bc = 32`, iterate over `(n, i, j)` tiles, map query heads to KV heads with `q_kv_heads_ratio = num_heads / num_kv_heads`, and accumulate in float32. Mask values should be applied in each tile before updating `m_i` and `l_i`.

When `mask == "causal"`, treat it as a block-level optimization opportunity: if a tile is fully invalid, skip that tile entirely; if a tile is fully valid, skip mask read/add for that tile and continue with matmul + online softmax. Also note that `L` and `S` are not always equal in causal attention, so do not hardcode logic that assumes `L == S`.

You can test your implementation by running:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k task_2
```

## Task 3: Implement `flash_attention` (GPU version)

{{#include copyright.md}}
