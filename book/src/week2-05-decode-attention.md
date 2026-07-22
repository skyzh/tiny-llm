# 🚧 Week 2 Day 6: Decode Attention

> 🚧 This newly introduced chapter is a work in progress.

During single-request decode, query length is normally one while the cached
key/value sequence grows by one token at a time. Week 1 expresses attention as
matrix multiplication, masking, softmax, and another matrix multiplication.
That is readable, but it materializes the complete score and probability rows.

In this chapter, keep the required path as a course-owned Python composition of
basic `mlx.core` array operations, then build an online-softmax Metal kernel and
measure whether it is actually faster. MLX still provides arrays, basic matmul,
streams, buffers, and extension dispatch; neither path calls an MLX-provided
scaled-dot-product-attention operator.

## Task 1: Preserve the Interface

Implement `scaled_dot_product_attention` in `week2_kernels.py` with these
model-facing shapes:

```plain
query: B, H_q,  L, D
key:   B, H_kv, S, D
value: B, H_kv, S, D
out:   B, H_q,  L, D
```

Validate that `H_q` is divisible by `H_kv`. Flatten batch and head dimensions
for the extension and map each query head to its shared KV head with:

```plain
kv_head = query_head / (H_q / H_kv)
```

Normalize explicit masks to `B * H_q, L, S`. Also pass a causal flag so the
kernel can skip future positions without constructing a causal-mask tensor.

For the required Python path, reshape query heads into `H_kv` groups and a
repeat dimension. Broadcasting then pairs several query heads with one KV head
without physically repeating the key and value tensors. Compute scaled scores,
apply the course's softmax helper, and multiply by value. This remains
easy to inspect and lets MLX choose efficient basic matmul kernels.

## Task 2: Try Online Softmax in Metal

Expose a separate `decode_attention_custom` function for the Metal experiment.
Assign four 32-lane SIMD groups to each query row. Each group visits every
fourth cached position; within a group:

1. Each lane multiplies a regularly spaced subset of query and key values.
2. `simd_sum` combines those partial dot products into one score.
3. Apply the scale, optional mask, and causal check.
4. Update a running maximum, softmax denominator, and weighted value
   accumulator.

The online update is:

```plain
new_max = max(running_max, score)
old_factor = exp(running_max - new_max)
score_factor = exp(score - new_max)
denominator = denominator * old_factor + score_factor
accumulator = accumulator * old_factor + score_factor * value
```

After its last cached position, each group writes its partial maximum,
denominator, and value accumulator to threadgroup memory. The first group
rescales and combines all four partial softmax states, then writes the final
output. Subtracting the maxima gives stable softmax without storing all `S`
scores or probabilities.

This removes two large intermediates and several dispatch boundaries from the
Week 1 graph. It is especially relevant as context grows: the avoided score and
probability tensors are proportional to `L * S`, while decode needs only the
final `D`-element result for each query head.

Load and store float16 or bfloat16 directly, but accumulate dot products,
softmax state, and weighted values in float32. Casting whole Q, K, and V tensors
outside the kernel creates extra dispatches and memory traffic; doing the
conversion in registers avoids that cost.

## Task 3: Integrate and Test

Route Week 2 attention through the faster of the two course-owned implementations
on the benchmark workload. The reference currently keeps the Python
composition as the required path and exposes the Metal kernel explicitly for
experimentation. Keep tiled prefill FlashAttention in Week 3; it solves a
different workload where both query and context lengths are large.

```bash
pdm run build-ext
pdm run test --week 2 --day 6
```

Test grouped-query head mapping, output shape, causal behavior, and explicit
masks against the readable Week 1 implementation. Use a tolerance because the
online softmax changes the floating-point reduction order.

## Expected Performance Contribution

**Estimated decode improvement: 0-5% at short context; longer context remains a
measurement exercise.** Quantized weight reads dominate short-context decode,
and a clear online-softmax kernel can still lose to the basic matmul composition
if it exposes too little parallel work. Do not claim the theoretical memory
saving as a speedup: report context length and measured throughput, and keep the
Python path when the custom kernel is slower.

{{#include copyright.md}}
