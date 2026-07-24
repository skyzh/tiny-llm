# 🚧 Week 2 Day 4: Decode Attention

> 🚧 This chapter is under review and may change.

This chapter starts only after profiling the quantized-matvec checkpoint across
cached contexts. Linear projections remain important, but the attention walk
grows with context while their shapes stay fixed. During single-request decode,
query length is normally one while the cached key/value sequence grows by one
token at a time. Week 1 expresses attention as matrix multiplication, masking,
softmax, and another matrix multiplication. That is readable, but it
materializes the complete score and probability rows.

First write a readable composition to preserve the equation, then replace its
matmuls and softmax with a course-owned online-softmax Metal kernel. Measure the
complete model before deciding whether to retain the dispatch. The kernel does
not call `mx.matmul` or an MLX-provided scaled-dot-product-attention
implementation; MLX still provides arrays, streams, buffers, and extension
dispatch.

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

As a readable intermediate step, reshape query heads into `H_kv` groups and a
repeat dimension. Broadcasting then pairs several query heads with one KV head
without physically repeating the key and value tensors. Express scaled scores,
softmax, and the weighted-value product explicitly. Use this form as a
correctness oracle and ablation, not as the completed optimized path: its
matmuls are MLX-provided operator implementations.

## Task 2: Implement Online Softmax in Metal

Expose `decode_attention_custom` for the Metal implementation. Cache the
scaled query fragment in registers before walking the cache; loading it again
for every key position is avoidable. Assign 32 32-lane SIMD groups to each
query row on the 128-192 token benchmark. Each group visits every 32nd cached
position; within a group:

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
denominator, and value accumulator to threadgroup memory. The first SIMD group
computes the common maximum and rescale factors. One thread computes the final
denominator, then the first `D` threads each combine one output dimension. This
keeps the final value reduction parallel across the head dimension.
Subtracting the maxima gives stable softmax without storing all `S` scores or
probabilities.

This removes two large intermediates and several dispatch boundaries from the
Week 1 graph. It is especially relevant as context grows: the avoided score and
probability tensors are proportional to `L * S`, while decode needs only the
final `D`-element result for each query head.

Load and store BF16 directly, but accumulate dot products,
softmax state, and weighted values in float32. Casting whole Q, K, and V tensors
outside the kernel creates extra dispatches and memory traffic; doing the
conversion in registers avoids that cost.

Use `fast::exp` for the rescale factors and compute each
factor once before applying it to the denominator and all value dimensions.
These ideas also appear in production vector-attention kernels, including MLX's
SDPA sources. The course kernel reimplements the algorithm and scheduling in
its own Metal code; it does not include or instantiate the MLX kernel.

### Scheduling Experiment

Compare eight, sixteen, and thirty-two SIMD groups with Qwen3-4B while holding
the context fixed. The number of groups is a workload parameter, not a
universal constant: more groups expose parallel score work but consume more
threads and threadgroup memory. Record the synchronized operator and
complete-model result for each schedule, then repeat the experiment when
context length changes.

## Task 3: Integrate and Measure

Route short-query, short-context Week 2 attention through the Metal
implementation. Dispatch back to the readable composition when the cached
context exceeds the measured crossover; a schedule that wins at 128 tokens
should not be forced onto 2,048 tokens. Retain the readable composition for
tests and ablations. Week 3 later combines this recurrence with paged K/V and
SIMD-matrix tiles for FlashAttention; prefill is a different workload where
both query and context lengths are large.

Set a concrete dispatch guard: use the course kernel only when query length is
at most eight and cached context length is at most 256. Otherwise use the
readable grouped-attention path. Keep this condition at the model call site so
the benchmarked operating range remains reviewable instead of becoming a
hidden performance policy inside the Metal kernel.

Keep arbitrary dense, per-request masks on the readable compatibility path.
They appear in the first continuous-batching exercise, while the normal Week 2
decode path uses no explicit mask. Week 3 replaces dense batch masks with paged
attention metadata instead of complicating this focused decode kernel.

```bash
pdm run build-ext
pdm run test --week 2 --day 4
```

Test grouped-query head mapping, output shape, causal behavior, and explicit
masks against the readable Week 1 implementation. Use a tolerance because the
online softmax changes the floating-point reduction order.

Run the preceding checkpoint and the model with the new dispatch under
otherwise identical settings:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2

pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint decode-attention --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

The model dispatches short-query contexts through the course kernel and falls
back to the exact readable Week 1 composition outside the validated range.
Reprofile the retained path. If repeated pointwise and reduction dispatches
become the largest remaining cluster, continue to Day 5; otherwise keep tuning
the dominant measured cost.

{{#include copyright.md}}
