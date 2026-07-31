# 🚧 Week 2 Day 5: Fused Decode Attention

> **Status: Experimental.** See the
> [Week 2 verification matrix](./week2-overview.md#verification-status) for
> what is continuously tested, locally measured, and still under review.

This chapter starts only after the Day 4 profile has verified that the fused
model kernels reduced the repeated pointwise cluster. Linear projections remain
important, but their operator latency is already close to the external
denominator, while the attention walk grows with cached context. During
single-request decode, query length is normally one while the cached key/value
sequence grows by one token at a time. Week 1 expresses attention as matrix
multiplication, masking, softmax, and another matrix multiplication. That is
readable, but it materializes the complete score and probability rows.

First write a readable composition to preserve the equation, then replace its
matmuls and softmax with an online-softmax Metal kernel in your solution.
Measure the complete model before deciding whether to retain the dispatch. The
kernel does not call `mx.matmul` or an MLX-provided
scaled-dot-product-attention
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
SDPA sources. Your kernel reimplements the algorithm and scheduling in
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

Set a concrete dispatch guard: use your Metal kernel only when query length is
at most eight and cached context length is at most 128. Otherwise use the
readable grouped-attention path. Keep this condition at the model call site so
the benchmarked operating range remains reviewable instead of becoming a
hidden performance policy inside the Metal kernel.

Keep arbitrary dense, per-request masks on the readable model path. The
primitive still accepts explicit masks so its arithmetic contract can be
tested, but the Week 2 dispatch guard selects the custom kernel only for
`None` or `"causal"`. Explicit masks appear in the first continuous-batching
exercise, while normal single-request decode uses no mask. Week 3 replaces
dense batch masks with paged-attention metadata instead of making them a hidden
performance policy in this focused model path.

```bash
pdm run build-ext
pdm run test --week 2 --day 5
```

Test grouped-query head mapping, output shape, causal behavior, and explicit
masks against the readable Week 1 implementation. The reference suite uses
Qwen's `D = 128`, query lengths 1 and 8, GQA ratios 1 and 4, and cached contexts
`1, 31, 32, 127, 128, 129, 255, 256`. It also checks both sides of the model's
`L <= 8` and `S <= 128` dispatch guard. Use a tolerance because online softmax
changes the floating-point reduction order.

Correctness over that grid does not prove that a fixed 32-SIMD-group schedule
is efficient. At contexts 1, 8, and 31, many of its 1,024 threads have no score
position to process. Run the same real-shape operator sweep on each target
machine before retaining the schedule:

```bash
for context in 1 31 32 127 128 129 255 256; do
  pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
    --section attention --context "${context}" \
    --query-length 1 --gqa-ratio 4 --attention-mask none
done

for context in 8 31 32 127 128 129 255 256; do
  pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
    --section attention --context "${context}" \
    --query-length 8 --gqa-ratio 4 --attention-mask causal
done
```

Repeat representative points with `--gqa-ratio 1` and
`--attention-mask explicit`. Keep M1 and M4 results as separate records; a
correctness run on the M1 CI runner is not evidence that the M4 crossover
applies there.

Run the preceding checkpoint and your solution with the new dispatch under
otherwise identical settings:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2

pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint decode-attention --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

Your solution dispatches short-query contexts through your Metal kernel and
falls back to the exact readable Week 1 composition outside the validated
range.

## Benchmark Analysis: Select Day 6

Measure the attention operator and the cumulative checkpoint separately, then
change the profile workload only after decode clears its target:

```bash
pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section attention --context 128

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-swiglu --variant week2-decode-attention --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case decode-attention:decode:128 \
  --case decode-attention:prefill:128 --warmup 4 --iterations 12
```

Repeat the attention microbenchmark at contexts 32, 128, 160, 192, and 256, and
attach that context sweep beside the `swiglu`/`decode-attention` model rows.
The intermediate points reveal whether the custom kernel has a useful measured
crossover rather than assuming that an endpoint applies to every context. Also
attach the decode and prefill kernel-group results. Reject the custom dispatch
if repeated fresh-process model runs do not improve, even when the isolated
kernel looks faster. If the operator wins only over a limited context range,
encode that measured crossover in the dispatch guard; replay a `.gputrace` with
`gpudebug` only when the operator and model results still disagree.

Once decode reaches 80% of MLX, read the prefill attribution as a new workload.
Continue to Day 6 only when quantized matrix-shaped projections dominate it.
The [reference checkpoint](./appendix-performance.md#day-5-fused-decode-attention)
pairs the context microbenchmarks, model delta, and prefill attribution that
select the matrix kernel.

{{#include copyright.md}}
