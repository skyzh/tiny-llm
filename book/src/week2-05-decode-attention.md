# 🚧 Week 2 Day 5: Fused Decode Attention

> **Status: Experimental.** See the
> [Week 2 verification matrix](./week2-overview.md#verification-status) for
> what is continuously tested, locally measured, and still under review.

This chapter starts only after the Day 4 evidence has verified that the fused
model kernels reduced the repeated pointwise cluster. Linear projections remain
important, but their operator latency is already close to the external
denominator, while attention is the next measured removable gap through cached
context `S <= 256`. Longer-context measurements use the Week 1 Python fallback for
caches beyond 256; every checked context through 256 uses the optimized path. During
single-request decode, query length is normally one while the cached key/value
sequence grows by one token at a time. Week 1 expresses attention as matrix
multiplication, masking, softmax, and another matrix multiplication. That is
expressed with `mlx.core`, but it materializes the complete score and probability rows.

First write a Python `mlx.core` composition to preserve the equation, then replace its
matmuls and softmax with an online-softmax Metal kernel in your solution.
Measure the complete model before deciding whether to retain the dispatch. The
kernel does not call `mx.matmul` or an MLX-provided
scaled-dot-product-attention
implementation; MLX still provides arrays, streams, buffers, and extension
dispatch.

## Task 1: Preserve the Interface

Modify `scaled_dot_product_attention` in
`src/tiny_llm/week2_kernels.py`. Keep this readable function as the oracle and
fallback; Task 2 modifies the separate `decode_attention_custom` entry point.

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

As a Python intermediate step, reshape query heads into `H_kv` groups and a
repeat dimension. Broadcasting then pairs several query heads with one KV head
without physically repeating the key and value tensors. Express scaled scores,
softmax, and the weighted-value product explicitly. Use this form as a
correctness oracle and ablation, not as the completed optimized path: its
matmuls are MLX-provided operator implementations.

## Task 2: Implement Online Softmax in Metal

Modify `tiny_llm_ext::decode_attention`,
`Week2DecodeAttention::eval_cpu`, and `Week2DecodeAttention::eval_gpu` in
`src/extensions/src/week2_kernels.cpp`, the `week2_decode_attention` function
in `src/extensions/src/week2_kernels.metal`, and `decode_attention_custom` in
`src/tiny_llm/week2_kernels.py`. The starter declaration, binding, source
stub, Metal file, and CMake registration are already present and labeled Week
2 Day 5; replace those fail-closed bodies rather than adding new names.

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

Modify `Qwen3MultiHeadAttention.__call__` in
`src/tiny_llm/qwen3_week2.py` to apply the measured dispatch guard. Keep
`scaled_dot_product_attention` as the explicit fallback and call
`decode_attention_custom` only inside the supported region.

Route short-query, short-context Week 2 attention through the Metal
implementation. Dispatch back to the Python `mlx.core` composition when the cached
context exceeds the measured crossover; a schedule that wins at 128 tokens
should not be forced onto 2,048 tokens. Retain the Python composition for
tests and ablations. Week 3 later combines this recurrence with paged K/V and
SIMD-matrix tiles for FlashAttention; prefill is a different workload where
both query and context lengths are large.

Set a concrete dispatch guard: use your Metal kernel only when query length is
at most two and cached context length is at most 256. Otherwise use the
Python grouped-attention path. Keep this condition at the model call site so
the benchmarked operating range remains reviewable instead of becoming a
hidden performance policy inside the Metal kernel.

Keep arbitrary dense, per-request masks on the Python model path. The
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
masks against the Python Week 1 implementation. The reference suite uses
Qwen's `D = 128`, query lengths 1 and 8, GQA ratios 1 and 4, and cached contexts
`1, 31, 32, 127, 128, 129, 255, 256`. It also checks both sides of the model's
`L <= 2` and `S <= 256` dispatch guard. Use a tolerance because online softmax
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
  --num-seqs 1 --min-input-len 32 --max-input-len 32 \
  --min-output-len 97 --max-output-len 97 --warmup 2 \
  --prefill-logits last

pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint decode-attention --model qwen3-4b \
  --num-seqs 1 --min-input-len 32 --max-input-len 32 \
  --min-output-len 97 --max-output-len 97 --warmup 2 \
  --prefill-logits last
```

Prefill produces the first token, so the 96 timed decode calls grow the cache
from `S=33` through `S=128`. Every one is inside the custom dispatch guard.
Your solution falls back to the exact Python Week 1 composition outside that
validated range.

## Benchmark Analysis: Verify Prefill Projections Are the Next Bottleneck

Measure the attention operator and the cumulative checkpoint separately. The
first progression is the matched short-context acceptance test for this
bounded kernel. The second keeps the fixed Week 2 denominator: its 128-token
prefill remains outside the query-length guard, while timed one-token decode
steps with `S=129` through `S=256` enter the current context guard:

```bash
pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section attention --context 32 --context 128 --context 160 \
  --context 192 --context 256 --context-repeats 6 \
  --json-output benchmark_results/week2-attention-context-sweep.json

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-swiglu --variant week2-decode-attention --variant mlx \
  --model qwen3-4b --input-len 32 --output-len 97 --warmup 2 \
  --prefill-logits last

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-swiglu --variant week2-decode-attention --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last

```

Repeat the attention microbenchmark at contexts 32, 128, 160, 192, and 256, and
attach that context sweep beside the short-context
`swiglu`/`decode-attention` model rows.
The intermediate points reveal whether the custom kernel has a useful measured
crossover rather than assuming that an endpoint applies to every context.
Reject the custom dispatch if repeated fresh-process short-context runs do not
improve, even when the isolated kernel looks faster. If the operator wins only
over a limited context range, encode that measured crossover in the dispatch
guard.

> **Optional profiling evidence.** Decode and prefill kernel-group results can
> explain how the workload divides its time, but they are reference evidence,
> not required output for this checkpoint.

The checked Qwen3-4B sweep on an M4 Pro used six forward/reverse context passes,
rotated every implementation order, and recorded all 60 samples per
implementation and pass:

| Context | Python reference | Metal | MLX | Metal speedup |
|---:|---:|---:|---:|---:|
| 32 | 143.0 us | 125.7 us | 116.3 us | 1.138x |
| 128 | 149.3 us | 136.3 us | 120.6 us | 1.095x |
| 160 | 151.2 us | 140.1 us | 120.9 us | 1.079x |
| 192 | 154.0 us | 143.9 us | 121.9 us | 1.071x |
| 256 | 158.0 us | 150.7 us | 122.8 us | 1.048x |

The Metal path wins at every measured point through 256, so 256 is the largest
evidenced context guard. The Python `mlx.core` path remains the policy beyond that
range; do not extrapolate the final 4.8% operator win to longer caches. The raw
record, including exact source SHA, model configuration, MLX and mlx-lm
versions, Metal compiler version, device information, execution order, samples,
and medians, is
`benchmark_results/m4-pro-qwen3-4b-week2-attention-context-sweep-mlx-0.32.0.json`.

The production-boundary sweep held context at 128, selected Qwen3-4B's 4:1 GQA
ratio, and balanced L1/L2/L4/L8 order over six passes. It used the causal form
for every query length: at `L=1` that mask permits the entire existing cache and
is equivalent to unmasked one-token decode, while `L>1` measures causal
multi-token chunks. Each pass also rotated the three implementation orders and
retained every sample:

| Query length | Python reference | Metal | MLX | Metal speedup | Pass wins |
|---:|---:|---:|---:|---:|---:|
| 1 | 244.4 us | 213.1 us | 155.9 us | 1.147x | 6/6 |
| 2 | 341.4 us | 258.8 us | 185.3 us | 1.319x | 6/6 |
| 4 | 322.7 us | 297.3 us | 197.4 us | 1.085x | 4/6 |
| 8 | 377.7 us | 491.5 us | 290.6 us | 0.768x | 0/6 |

L4's aggregate median improved, but it lost two of six balanced passes. L2 is
the largest repeat-consistent win, so the dispatch guard remains conservative
at `L <= 2`; L4 and L8 use the Python path. Reproduce the recorded sweep with:

```bash
pdm run bench-week2-operators --solution tiny_llm_ref --model qwen3-4b \
  --section attention --context 128 \
  --query-length 1 --query-length 2 --query-length 4 --query-length 8 \
  --gqa-ratio 4 --attention-mask causal --context-repeats 6 \
  --warmup 12 --iterations 60 \
  --json-output benchmark_results/week2-attention-query-sweep.json
```

The checked raw record is
`benchmark_results/m4-pro-qwen3-4b-week2-attention-query-sweep-mlx-0.32.0.json`.

In the fixed `128/129` workload, prefill has `L=128` and uses the Python path.
The first timed decode call appends the new token before the guard sees `S=129`;
the one-token decode calls through `S=256` therefore use the custom path. Keep
the fixed workload separate from the short-context acceptance run. Continue to
Day 6 after the correctness tests pass, the direct source trace proves the
bounded attention dispatch and its fallback, repeated short-context runs retain
the gain, and the fixed `128/129` control confirms that prefill is unchanged and
still routes through Day 3's matrix-shaped projection path.

> **Optional profiling evidence.** The
> [reference checkpoint](./appendix-performance.md#day-5-fused-decode-attention)
> pairs the context sweep, short-context model delta, and fixed-workload
> control with a separate prefill attribution. The attribution explains why the
> course targets matrix-shaped projections next; it is not a prerequisite for
> Day 6.

{{#include copyright.md}}
