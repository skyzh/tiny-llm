# 🚧 Week 3 Day 7 (Optional): Performance Lab

> 🚧 This chapter is under review and may change.

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

### Hoist Quantization Parameters at the Group Boundary

The weights use 4-bit values with one scale and bias for every 128 reduction
elements. An 8×8 matrix kernel advances through that group in sixteen
eight-wide tiles. Loading the same scale and bias inside every tile repeats
parameter-address calculation and device reads sixteen times for each output
fragment.

Loop over quantization groups first. Each lane loads the scale and bias for
its two output-fragment elements into registers, then reuses them while it
unpacks all sixteen matrix tiles in that group:

```plain
for each 128-value quantization group:
    load scale and bias for this output fragment
    for each of its sixteen 8-value reduction tiles:
        unpack and dequantize weights with the registered parameters
        matrix-multiply-accumulate into FP32
```

This changes scheduling, not the quantization equation or model format. In the
matched Qwen3-0.6B measurement it reduced the synchronized 128-token Q
projection from about 370 to 328 µs. Experiments that computed two output
tiles per SIMD group, broadcast packed weights with shuffles, or reduced the
threadgroup from eight to four SIMD groups all regressed; the optimization
ledger records those results so they are not rediscovered as assumed wins.

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
at least 70% of MLX decode throughput on the same machine. Week 3 adds paging,
allocator, and scheduler work around a paged decode kernel that is now close to
the dense course attention operator; do not reuse the Week 2 single-request
acceptance ratio as the serving-engine target. Measure prefill, aggregate
request throughput, cache capacity, and page reuse separately. The lab may not
replace a slow operator with the matching MLX implementation; profile and
report each remaining gap honestly.

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

Create an experiment ledger as you work: record the hypothesis, synchronized
measurement, correctness result, and keep/reject decision. Use the reference
ledger below to select reproducible scheduling exercises, not as a substitute
for measurements on your hardware. End-to-end effects overlap, so never add
the percentages across rows.

| Change | Decision | Representative effect and reason |
|---|---|---|
| Keep weights quantized | Keep | Several-fold decode gain over Week 1 by reducing weight traffic. |
| Two-column SIMD matvec | Keep | About +5.7% in its first isolated end-to-end ablation; shares activation loads across columns. |
| Eight-column vocabulary matvec | Keep | Best measured layout for the very wide tied output head; smaller projections use two columns to avoid register pressure. |
| Remove matvec activation barrier | Keep | Roughly 238.6 → 247-252 tok/s; a cache-hot 2 KB vector was cheaper to reread than to copy and synchronize in every threadgroup. |
| Vanilla → `simdgroup_matrix` prefill | Keep | About 1,005 → 2,052 prefill tok/s; 8×8 matrix fragments replace scalar dot products. |
| Hoist quantization parameters per group | Keep | Scale and bias are constant across sixteen 8-wide reduction tiles. Keeping them in registers improved the 128-token Q projection from about 370 to 328 µs and the matched lab checkpoint to 2,371 prefill tok/s. |
| Direct quantized embedding | Keep | Roughly 20% lower isolated latency in a representative run, about 1% end to end; gathers and dequantizes without intermediate arrays. |
| 256-thread RMSNorm | Keep | Closes most of the gap left by a one-SIMD-group reduction. |
| Four-head RoPE reuse | Keep | Avoids repeated angle/sine/cosine work; current isolated latency is within roughly 7-14% of MLX. |
| Fused SwiGLU | Keep | One dispatch and no intermediates; approximately equal to MLX in isolated measurements. |
| 32-group online decode attention | Keep | Owns the complete attention algorithm; 8 and 16 groups exposed less parallel work on the course context. |
| No decode causal-mask graph | Keep | Pass `None` when `L = 1`; every cached position is already valid. |
| Normalize RoPE offsets once | Keep | Builds one batch offset array per model call rather than once per layer. |
| Last-token logits | Keep | Avoids vocabulary projection for prompt positions that generation never samples. |
| Zero-gather FlashAttention prefill | Keep | Runs Day 2 FlashAttention directly on fresh K/V before paged decode. With the SIMD-matrix lab it improved matched prefill about 1.7% at 128 tokens and 2.0% at 512; decode was unchanged. |
| Fuse Q/K/V and gate/up projections | Reject | About 242 → 227 tok/s; fewer Python calls did not offset a more complex kernel and dispatch map. |
| Concatenate quantized weights at runtime | Reject | About 235 → 217 tok/s; constructing larger arrays added work and memory traffic. |
| Preallocate chunked dense KV cache | Reject | About 235 → 229 tok/s; strided logical slices hurt the following operations. Paging belongs in Week 3. |
| Share 32×16 prefill tiles in threadgroup memory | Reject | About 2,052 → 1,848 prefill tok/s; barriers outweighed reduced global loads at 128 tokens. |
| Two output tiles per SIMD group | Reject | About 328 → 338 µs for the 128-token Q projection; reusing the activation fragment did not offset the second FP32 accumulator's register pressure. |
| Broadcast packed weights within a SIMD group | Reject | About 370 → 699 µs for the 128-token Q projection; coalesced cached loads were cheaper than repeated SIMD shuffles. |
| Four SIMD groups per prefill threadgroup | Reject | About 370 → 385 µs for the same projection; the smaller scheduling unit added launches without improving occupancy. |
| Broadcast scale/bias within the wide matvec | Reject | Increased vocabulary-head latency; extra loop structure and broadcasts outweighed parameter-load savings. |
| Four-output ordinary matvec | Reject | About 249 → 232 tok/s; extra register pressure outweighed activation reuse. |
| Affine identity in the ordinary matvec | Reject | About 249 → 244.5 tok/s; fewer arithmetic instructions did not improve the executed schedule. |
| Four SIMD groups per matvec threadgroup | Reject | Two runs near 248.5 tok/s; smaller threadgroups added scheduling overhead without useful new parallelism. |
| Sixteen groups for the vocabulary matvec | Reject | About 250.6 → 247.2 tok/s; a 512-thread scheduling unit offset the reduction in threadgroup count. |

This ledger is also a scheduling lesson. Fewer loads, fewer graph nodes, or
fewer dispatches are hypotheses. Only synchronized end-to-end measurement says
whether their occupancy, barrier, register, and layout tradeoffs helped.

{{#include copyright.md}}
