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

Then implement `FlashAttention::eval_cpu(...)` with tiled online softmax. Use `Br = 32` and `Bc = 32`, and the rationale for this choice will be explained in the GPU section. Iterate over `(n, i, j)` tiles, map query heads to KV heads with `q_kv_heads_ratio = num_heads / num_kv_heads`, and accumulate in float32. Mask values should be applied in each tile before updating `m_i` and `l_i`.

When `mask == "causal"`, treat it as a block-level optimization opportunity: if a tile is fully invalid, skip that tile entirely; if a tile is fully valid, skip mask read/add for that tile and continue with matmul + online softmax. Also note that `L` and `S` are not always equal in causal attention, so do not hardcode logic that assumes `L == S`.

You can test your implementation by running:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k task_2
```

## Task 3: Implement `flash_attention` (GPU version)

```
src/extensions/src/flash_attention.metal
src/extensions/src/flash_attention.cpp
src/extensions/CMakeLists.txt
```

Now implement the GPU path for the same algorithm.

### GPU Parallelization Strategy

The key to an efficient GPU implementation is understanding how to map the tiled algorithm to Metal's execution model.

#### Why Br = 32 and Bc = 32?

The tile sizes are not arbitrary—they are constrained by Apple GPU hardware:

| Constraint | Source | Value |
|------------|--------|-------|
| SIMD width | Apple GPU fixed | 32 |
| Max threads per threadgroup | Hardware limit | 1024 |
| Bc | = SIMD width (for efficient `simd_sum`/`simd_max`) | 32 |
| Br | = 1024 / 32 | 32 |
| Threadgroup memory | 32KB limit | Fits `q_local[32][128]` + `o_i[32][128]` |

With Br=32 and Bc=32, we get 32×32 = 1024 threads per threadgroup, which exactly fills the hardware limit.

#### Grid and Threadgroup Layout

```plain
Grid (num_threadgroups):
┌───────────────────────┬───────────────────────┬───────────────────────┐
│ TG(0, 0)              │ TG(1, 0)              │ TG(2, 0)              │
│ head=0, qtile=0       │ head=1, qtile=0       │ head=2, qtile=0       │
├───────────────────────┼───────────────────────┼───────────────────────┤
│ TG(0, 1)              │ TG(1, 1)              │ TG(2, 1)              │
│ head=0, qtile=1       │ head=1, qtile=1       │ head=2, qtile=1       │
├───────────────────────┼───────────────────────┼───────────────────────┤
│ ...                   │ ...                   │ ...                   │
└───────────────────────┴───────────────────────┴───────────────────────┘
     X: N (heads)         Y: Tr (query blocks)
```

Each threadgroup is responsible for one `(head, Q-tile)` output block.

#### Thread Mapping Within a Threadgroup

Each threadgroup handles one Q block (size Br×E) for one head. Inside the threadgroup:

```plain
Threadgroup = 32 SIMD groups × 32 threads/group = 1024 threads

┌────────────────────────────────────────────────┐
│ SIMD group 0  → Q[0, :]  (handles row 0)       │ ← 32 threads
│ SIMD group 1  → Q[1, :]  (handles row 1)       │ ← 32 threads
│ SIMD group 2  → Q[2, :]  (handles row 2)       │ ← 32 threads
│ ...                                             │
│ SIMD group 31 → Q[31, :] (handles row 31)      │ ← 32 threads
└────────────────────────────────────────────────┘
```

Inside that single threadgroup, the kernel runs a **serial** loop over all K/V tiles `j = 0..Tc-1`.

#### Computing S = Q @ K^T

Each thread computes one element of the 32×32 score matrix. Here's how the matrix multiplication maps to threads:

```plain
Q block [Br=32, E=128]              K^T [E=128, Bc=32]
┌───────────────────────┐           ┌───┬───┬───┬─...─┬───┐
│ Q[0,:]  (128 elements)│           │   │   │   │     │   │
├───────────────────────┤           │ K │ K │ K │     │ K │
│ Q[1,:]                │           │[0]│[1]│[2]│ ... │[31]│
├───────────────────────┤     @     │ T │ T │ T │     │ T │
│ Q[2,:]                │           │   │   │   │     │   │
├───────────────────────┤           │128│128│128│     │128│
│ ...                   │           │   │   │   │     │   │
├───────────────────────┤           │   │   │   │     │   │
│ Q[31,:]               │           │   │   │   │     │   │
└───────────────────────┘           └───┴───┴───┴─...─┴───┘
        ↑                                 ↑
   simd_gid = a                      simd_lid = b
   (which row)                       (which column)
```

Result: S block [Br=32, Bc=32], each element computed by one thread:

```plain
                    simd_lid (b)
              0     1     2    ...   31
            ┌─────┬─────┬─────┬─...─┬─────┐
          0 │S0,0 │S0,1 │S0,2 │     │S0,31│  ← SIMD group 0 (32 threads)
            ├─────┼─────┼─────┼─...─┼─────┤
simd_gid  1 │S1,0 │S1,1 │S1,2 │     │S1,31│  ← SIMD group 1
  (a)       ├─────┼─────┼─────┼─...─┼─────┤
          2 │S2,0 │S2,1 │S2,2 │     │S2,31│  ← SIMD group 2
            ├─────┼─────┼─────┼─...─┼─────┤
        ... │ ... │ ... │ ... │     │ ... │
            ├─────┼─────┼─────┼─...─┼─────┤
         31 │S31,0│S31,1│S31,2│     │S31,31│ ← SIMD group 31
            └─────┴─────┴─────┴─...─┴─────┘

Thread (a=2, b=5) computes:
  S[2,5] = Q[2,0]*K[5,0] + Q[2,1]*K[5,1] + ... + Q[2,127]*K[5,127]
         = dot product of Q row 2 with K row 5 (128 multiply-adds)
```

After computing S[a,b], each thread holds one attention score. Row-wise reductions use SIMD intrinsics—all 32 threads in the same SIMD group cooperate:

```plain
SIMD group 2 (threads with simd_gid=2):
  Thread b=0 has S[2,0]
  Thread b=1 has S[2,1]
  ...
  Thread b=31 has S[2,31]

  simd_max(s_a_b) → all 32 threads get max(S[2,0], S[2,1], ..., S[2,31])
  simd_sum(p_a_b) → all 32 threads get sum(P[2,0], P[2,1], ..., P[2,31])
```

```metal
float rowmax = simd_max(s_a_b);  // max across 32 threads in same SIMD group
float rowsum = simd_sum(p_a_b);  // sum across 32 threads in same SIMD group
```

#### Computing O = P @ V inside a SIMD group

After softmax, we need to accumulate the output tile. A natural first thought is: "Can we assign threads to output elements the same way we did for S = Q @ K^T?" The answer is **no**, because the output dimensions don't match:

```plain
Q @ K^T:                         P @ V:
┌─────────┐   ┌─────────┐       ┌─────────┐   ┌─────────────────┐
│ Q       │   │ K^T     │       │ P       │   │ V               │
│[Br, E]  │ @ │[E, Bc]  │       │[Br, Bc] │ @ │[Bc, E]          │
│[32,128] │   │[128,32] │       │[32, 32] │   │[32, 128]        │
└─────────┘   └─────────┘       └─────────┘   └─────────────────┘
         ↓                               ↓
   S [Br, Bc]                      O [Br, E]
   [32, 32]                        [32, 128]
   = 1024 elements                 = 4096 elements
        ↓                               ↓
   1024 threads ✓                  1024 threads ✗
   (one per element)               (not enough!)
```

For S = Q @ K^T, we have 1024 output elements and 1024 threads—perfect one-to-one mapping. But for O = P @ V, we have 4096 output elements but only 1024 threads. The mismatch comes from the embedding dimension: **E = 128 ≠ Bc = 32**.

So we use a different strategy: instead of assigning threads to output columns, we **loop over the 128 output columns** and use SIMD reduction for each:

```plain
For each output element O[a, c]:
  
  O[a, c] = sum over b: P[a, b] * V[b, c]
            └───────────────────────────┘
                   32 terms (Bc = 32)
                         ↓
            simd_sum can handle this!

  Thread assignment:
    - simd_gid = a (which output row)
    - simd_lid = b (which term in the sum)
    
  Code:
    for c in 0..E-1:                      // loop 128 times
        val = P[a, b] * V[b, c]           // each lane computes one term
        result = simd_sum(val)            // reduce 32 terms → 1 result
        if simd_lid == 0:
            o_i[a, c] += result           // only lane 0 writes
```

The key insight: even though we can't parallelize over the E dimension (because E > SIMD width), we **can** parallelize the reduction over Bc = 32, which matches SIMD width exactly.

#### Memory Hierarchy

```plain
┌─────────────────────────────────────────────────────────┐
│ Global Memory (HBM)                                     │
│ Q[N, L, E], K[N_kv, S, E], V[N_kv, S, E]               │
└─────────────────────────────────────────────────────────┘
                    ↓ load once per Q block
┌─────────────────────────────────────────────────────────┐
│ Threadgroup Memory (SRAM, 32KB)                         │
│ q_local[Br][E]  ← Q block, reused for all Tc iterations │
│ o_i[Br][E]      ← accumulated output                    │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Registers (per thread)                                  │
│ m_i, l_i, s_a_b, p_a_b                                  │
└─────────────────────────────────────────────────────────┘
```

K and V blocks are streamed from global memory in the inner loop over Tc. The Q block is loaded once into threadgroup memory and reused across all K/V tiles.

### Implementation

In `flash_attention.metal`, write `flash_attention_f32_e128` with one threadgroup per `(n, i)` tile, where `n` is the flattened head batch and `i` is the query tile index. Use threadgroup memory for local Q and partial O, and use SIMD reductions (`simd_max`, `simd_sum`) for row-wise max/sum updates.

In `eval_gpu(...)`, load the kernel from the extension, bind inputs/outputs and scalar constants (`N`, `L`, `S`, `E`, head counts, `scale`, tile sizes), and dispatch over `(N, Tr, 1)`. Keep the same contiguous checks as CPU path. Also remember to add `src/flash_attention.metal` into `mlx_build_metallib(...)` in `CMakeLists.txt`.

You can test your implementation by running:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k task_3
```

## Task 4: Model Integration

```
src/tiny_llm/qwen2_week2.py
```

Finally, wire the kernel into model execution. Keep the existing grouped attention path as fallback, add the `use_flash_attention` switch in `Qwen2MultiHeadAttention`, and propagate `enable_flash_attn` from model initialization into each block. After KV cache update, build the correct causal mask for `L x S`, run attention in float32, and cast back to activation dtype.

You can run generation with Flash Attention enabled:

```bash
pdm run main --solution tiny_llm --loader week2 --model qwen2-0.5b --enable-flash-attn
```

You can also benchmark throughput with and without Flash Attention:

```bash
pdm bench --solution tiny_llm --loader week2 --model qwen2-0.5b
pdm bench --solution tiny_llm --loader week2 --model qwen2-0.5b --enable-flash-attn
```

{{#include copyright.md}}
