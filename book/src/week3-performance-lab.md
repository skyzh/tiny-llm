# 🚧 Week 3 Performance Lab

> 🚧 This newly introduced chapter is a work in progress.

This optional lab starts from the minimal Week 2 decode model and explores
optimizations that are useful but not required to understand that model. It is
the right place for prefill matrix tiling, specialized embedding kernels,
last-token logits, and end-to-end scheduling experiments.

## Accelerate Prefill Matrix Products

Week 2 deliberately keeps the scalar quantized matrix-matrix kernel for
prefill so its required path contains only one optimized matrix-vector layout.
In Week 3, add an 8×8 `simdgroup_matrix` kernel for inputs with more than eight
rows. Load fragments cooperatively, perform the matrix multiply in hardware,
and store the tile with boundary checks. The Week 3 model opts into this path
through `QuantizedWeights.use_simdgroup_matmul`; the Week 2 model does not.

Each SIMD group owns one 8×8 output tile. For every eight-wide reduction step,
its 32 lanes load an activation fragment, unpack and dequantize a weight
fragment, issue `simdgroup_multiply_accumulate`, and retain a float32
accumulator fragment until the final store. Keeping only the input fragments in
16-bit precision matters: a 16-bit accumulator passed an isolated loose test
but accumulated visible error across transformer layers.

The kernel borrows the scheduling ideas behind MLX Steel without instantiating
Steel templates or calling MLX operator kernels. An attempted 32×16 tile copied
shared A/B blocks through threadgroup memory. It reduced global loads but added
two barriers per reduction tile and regressed the 128-token M1 Pro prefill from
about 2,052 to 1,848 tok/s. Reuse is valuable only when it costs less than its
synchronization and occupancy loss.

This is also where a direct quantized embedding kernel belongs. Fuse row
gather, int4 unpacking, and affine dequantization into one dispatch, then opt
the Week 3 embedding into it. It is a small end-to-end win, so keeping it out
of Week 2 makes the earlier lesson easier to follow. Map one thread to one
requested embedding element and accept both int32 prompt IDs and uint32 sampled
IDs so the fused path does not create a token-cast graph node.

The API boundary makes the experiment visible. `Qwen3ModelWeek3` applies these
settings when it receives `enable_performance_lab=True`:

```python
# Week 2: vanilla prefill matmul and readable embedding dequantization.
weights = QuantizedWeights.from_mlx_layer(layer)
embedding = QuantizedEmbedding(vocab_size, hidden_size, weights)

# Week 3: explicitly opt into both advanced paths.
weights = QuantizedWeights.from_mlx_layer(
    layer, use_simdgroup_matmul=True
)
embedding = QuantizedEmbedding(
    vocab_size, hidden_size, weights, use_custom_kernel=True
)
```

## Keep Only Needed Logits

Prefill computes hidden states for the whole prompt, but generation samples only
from the final position. The shared Week 2/3 model interface exposes an optional
`logits_to_keep` argument. In this lab, make the Week 3 serving callers request
one position and verify that the model slices hidden states before the final
norm and vocabulary projection:

```python
if logits_to_keep is not None:
    h = h[:, -logits_to_keep:, :]
```

Generation and batching should request one position. Callers that need all
positions, including correctness tests and prompt scoring, pass `None`.

## Preserve the Incremental Layers

- `qwen3_week1.py` keeps readable Python operators and dequantized weights.
- `qwen3_week2.py` uses quantized weights and `week2_kernels.py`.
- `qwen3_week3.py` uses the Week 2 optimization interfaces, reuses the cache
  contract, and adds paged storage and serving mechanisms.

Later weeks must not overwrite earlier files. The starter and reference trees
expose the same module and function names so each checkpoint remains runnable.

## Validate Correctness

```bash
pdm run test --week 3 --day 7
```

Check full-sequence logits with `logits_to_keep=None`, last-position logits with
`logits_to_keep=1`, and incremental decoding against the MLX model.

## Measure the Result

Repeat the matched benchmark from Week 2. The minimal Week 2 path should reach
at least 70% of MLX decode throughput on the same machine. Week 3 adds a slower
teaching paged-attention path, so do not reuse that single-request acceptance
ratio as a serving-engine target. Measure prefill, aggregate request
throughput, cache capacity, and page reuse separately. The lab may not replace
a slow operator with the matching MLX implementation; profile and report each
remaining gap honestly.

```bash
pdm run bench --solution tiny_llm_ref --loader week3 \
  --enable-performance-lab --model qwen3-0.6b
```

Measure the serving engine separately with the continuous-batching mode. This
reports aggregate output and decode throughput rather than treating one
request's latency as the Week 3 result:

```bash
pdm run bench --solution tiny_llm_ref --loader week3 --batch-decode \
  --num-seqs 16 --batch-size 4 --model qwen3-0.6b
```

Before accepting the result, inspect the Week 2 source for accidental shortcuts.
The model may use `mlx_lm` for loading and `mlx.core` for arrays, device
execution, synchronization, and extension dispatch, but its learned operators
must resolve to the Python/C++/Metal implementations from this course.

## Optimization Ledger

Record both retained and rejected experiments. End-to-end effects overlap, so
the table uses representative ranges rather than pretending every row is
independently additive.

| Change | Decision | Representative effect and reason |
|---|---|---|
| Keep weights quantized | Keep | Several-fold decode gain over Week 1 by reducing weight traffic. |
| Two-column SIMD matvec | Keep | About +5.7% in its first isolated end-to-end ablation; shares activation loads across columns. |
| Eight-column vocabulary matvec | Keep | Best measured layout for the very wide tied output head; smaller projections use two columns to avoid register pressure. |
| Remove matvec activation barrier | Keep | Roughly 238.6 → 247-252 tok/s; a cache-hot 2 KB vector was cheaper to reread than to copy and synchronize in every threadgroup. |
| Vanilla → `simdgroup_matrix` prefill | Keep | About 1,005 → 2,052 prefill tok/s; 8×8 matrix fragments replace scalar dot products. |
| Direct quantized embedding | Keep | Roughly 20% lower isolated latency in a representative run, about 1% end to end; gathers and dequantizes without intermediate arrays. |
| 256-thread RMSNorm | Keep | Closes most of the gap left by a one-SIMD-group reduction. |
| Four-head RoPE reuse | Keep | Avoids repeated angle/sine/cosine work; current isolated latency is within roughly 7-14% of MLX. |
| Fused SwiGLU | Keep | One dispatch and no intermediates; approximately equal to MLX in isolated measurements. |
| 32-group online decode attention | Keep | Owns the complete attention algorithm; 8 and 16 groups exposed less parallel work on the course context. |
| No decode causal-mask graph | Keep | Pass `None` when `L = 1`; every cached position is already valid. |
| Normalize RoPE offsets once | Keep | Builds one batch offset array per model call rather than once per layer. |
| Last-token logits | Keep | Avoids vocabulary projection for prompt positions that generation never samples. |
| Fuse Q/K/V and gate/up projections | Reject | About 242 → 227 tok/s; fewer Python calls did not offset a more complex kernel and dispatch map. |
| Concatenate quantized weights at runtime | Reject | About 235 → 217 tok/s; constructing larger arrays added work and memory traffic. |
| Preallocate chunked dense KV cache | Reject | About 235 → 229 tok/s; strided logical slices hurt the following operations. Paging belongs in Week 3. |
| Share 32×16 prefill tiles in threadgroup memory | Reject | About 2,052 → 1,848 prefill tok/s; barriers outweighed reduced global loads at 128 tokens. |
| Broadcast scale/bias within the wide matvec | Reject | Increased vocabulary-head latency; extra loop structure and broadcasts outweighed parameter-load savings. |
| Four-output ordinary matvec | Reject | About 249 → 232 tok/s; extra register pressure outweighed activation reuse. |
| Affine identity in the ordinary matvec | Reject | About 249 → 244.5 tok/s; fewer arithmetic instructions did not improve the executed schedule. |
| Four SIMD groups per matvec threadgroup | Reject | Two runs near 248.5 tok/s; smaller threadgroups added scheduling overhead without useful new parallelism. |
| Sixteen groups for the vocabulary matvec | Reject | About 250.6 → 247.2 tok/s; a 512-thread scheduling unit offset the reduction in threadgroup count. |

This ledger is also a scheduling lesson. Fewer loads, fewer graph nodes, or
fewer dispatches are hypotheses. Only synchronized end-to-end measurement says
whether their occupancy, barrier, register, and layout tradeoffs helped.

## Expected Performance Contribution

**Measured prefill improvement from last-token logits: about 40x for the output
projection alone and 1.29x for the complete model on Qwen3-0.6B on an M4 Pro.**
Normalizing RoPE offsets once saved about 2% per isolated call. Omitting the
single-token causal flag was not measurable. Graph cleanup avoids work outside
the transformer blocks, but small changes below the run-to-run noise floor
must be reported as such.

On that same M4 Pro, the performance-lab path based on Week 2 improved decode
from about 19.4 tok/s for Week 1 to about 246 tok/s, or about **12.7x**. Its
prefill measured about 2,042 tok/s versus about 3,318 for the dense Week 1
model, because the educational quantized prefill kernel did not win this
compute-dense shape. The stable matched MLX decode was about 320 tok/s, putting
the dense course path at 76.8% of MLX on this machine.

One-factor cached-decode ablations give the following attribution. Each row
uses two models with the same loaded weights, alternates optimized and vanilla
runs, and changes only the named component:

| Replacement | Vanilla tok/s | Optimized tok/s | Throughput gain |
|---|---:|---:|---:|
| Quantized embedding gather | 243.87 | 245.75 | +0.8% |
| RMSNorm Metal kernel | 185.28 | 246.16 | +32.9% |
| RoPE Metal kernel | 193.38 | 245.92 | +27.2% |
| Fused SwiGLU | 219.14 | 245.29 | +11.9% |
| Online decode attention | 245.17 | 245.73 | +0.2% |

These rows are reverse ablations from the finished model and are not additive.
As a diagnostic only, temporarily replacing the course QMV with MLX improved
throughput by 14.5%, replacing attention improved it by 9.8%, and replacing
both reached about 309.7 tok/s. Those substitutions are not part of the solution;
they identify the two course-owned kernels with the largest remaining ceiling.

{{#include copyright.md}}
