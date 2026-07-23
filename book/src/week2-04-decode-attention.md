# 🚧 Week 2 Day 4: Decode Attention

> 🚧 This newly introduced chapter is a work in progress.

This chapter starts from the quantized-matvec checkpoint. During
single-request decode, query length is normally one while the cached
key/value sequence grows by one token at a time. Week 1 expresses attention as
matrix multiplication, masking, softmax, and another matrix multiplication.
That is readable, but it materializes the complete score and probability rows.

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

## Task 2: Try Online Softmax in Metal

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
denominator, then the first `D` threads combine one output dimension each. This
parallel final reduction was faster than making the first 32 lanes each reduce
four dimensions. Subtracting the maxima gives stable softmax without storing
all `S` scores or probabilities.

This removes two large intermediates and several dispatch boundaries from the
Week 1 graph. It is especially relevant as context grows: the avoided score and
probability tensors are proportional to `L * S`, while decode needs only the
final `D`-element result for each query head.

Load and store float16 or bfloat16 directly, but accumulate dot products,
softmax state, and weighted values in float32. Casting whole Q, K, and V tensors
outside the kernel creates extra dispatches and memory traffic; doing the
conversion in registers avoids that cost.

The implementation uses `fast::exp` for the rescale factors and computes each
factor once before applying it to the denominator and all value dimensions.
These ideas also appear in production vector-attention kernels, including MLX's
SDPA sources. The course kernel reimplements the algorithm and scheduling in
its own Metal code; it does not include or instantiate the MLX kernel.

### Scheduling Ablation

The number of SIMD groups is a workload parameter, not a universal constant.
On the M1 Pro benchmark, eight groups reduced scratch memory but serialized too
many cache positions and achieved only about 215 decode tok/s. Sixteen groups
reached roughly 232 tok/s. Thirty-two groups plus the parallel final reduction
reached roughly 238-239 tok/s in the historical completed-model ablation. More
groups increase parallel score work but also consume more threads and
threadgroup memory; measure again when context length or head dimension
changes.

## Task 3: Integrate and Measure

Route short-query, short-context Week 2 attention through the Metal
implementation. Dispatch back to the readable composition when the cached
context exceeds the measured crossover; a schedule that wins at 128 tokens
should not be forced onto 2,048 tokens. Retain the readable composition for
tests and ablations, and retain tiled prefill FlashAttention in Week 3; prefill
is a different workload where both query and context lengths are large.

The retained reference guard is deliberately concrete: use the course kernel
only when query length is at most eight and cached context length is at most
256. Otherwise use the readable grouped-attention path. Keeping this condition
at the model call site makes the benchmarked operating range reviewable instead
of burying a performance policy inside the Metal kernel.

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
  --week2-checkpoint quantized-matvec --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2

pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint decode-attention --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

The model dispatches short-query contexts through the course kernel and falls
back to the exact readable Week 1 composition outside the measured range. This
is the same evidence-driven decision used for tile sizes, barriers, and
threadgroup layouts elsewhere in the course.

## Expected Performance Contribution

**Measured cumulative improvement: 134.24 to 143.42 tok/s (+6.8%) on an M4
Pro at context 128.** The resulting checkpoint is 7.35x Week 1 and 56.3% below
matched MLX. Against an alternative readable bfloat16 composition, the same
operator was about 7% faster at context 128 but slower at longer contexts;
that comparison is not the course ladder because the preceding checkpoint
deliberately preserves Week 1's float32 equation. Quantized weight reads still
dominate short-context decode, and the fixed 32-group schedule does not scale
monotonically. Do not claim the theoretical memory saving as a speedup: report
dtype, context length, schedule, and measured end-to-end throughput.

{{#include copyright.md}}
