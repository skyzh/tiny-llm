# 🚧 Week 2 Day 5: Tiled Dense Prefill Attention

Complete [Day 4: Fused Model Primitives](./week2-04-fused-model-kernels.md)
first. You have a cached, packed-W4 Qwen3 model whose `swiglu` checkpoint
already includes SIMD matrix prefill, RMSNorm, RoPE, and SwiGLU. Day 5 changes
the dense attention computation for a prompt with at least nine query tokens.
One-token decode keeps readable grouped attention.

Keep the same cached 0.6B model and request while you work. Save the Day 4
`swiglu` product result before editing attention; later compare it with
`tiled-prefill` and the completed `selected` model under that same workload.
The progression runner uses fresh processes and balanced order. Its `--offline`
mode needs the model files cached already:

```bash
pdm run build-ext
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-swiglu --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day5-control.json
```

The `mlx` row is a separate full-model baseline. Your course model and its
cache remain the subject of the Day 5 work. Build the supplied reference
extension and run its completed Day 5 checks to see the target behavior; the
learner extension retains its own TODOs:

```bash
pdm run build-ext-ref
pdm run test-refsol --week 2 --day 5
```

## Task 1: Make Tiled Attention Match the Readable Result

The supplied `dense_prefill_attention_mma` adapter in
`src/tiny_llm/week2_kernels.py` prepares the model-facing arrays and calls
the learner-owned private `tiny_llm_ext::_dense_attention_prefill_mma` binding.
Implement that binding and `Week2DensePrefillMMA::eval_gpu` in
`src/extensions/src/week2_kernels.cpp`, plus
`week2_dense_prefill_mma_bf16_d128` in
`src/extensions/src/week2_kernels.metal`. Keep the supplied Python adapter and
header interface. Replace the `Week2DensePrefillMMA::eval_cpu` starter TODO in
`src/extensions/src/week2_kernels.cpp` with a GPU-only error. The adapter
validates the shape, prepares contiguous inputs
and an optional mask, invokes the native primitive, and restores the
model-facing shape.

The operator accepts grouped-query attention in this layout:

```text
Q:      B, Hq,  L, 128
K, V:   B, Hkv, S, 128
output: B, Hq,  L, 128
```

Here `L` is the new prompt length and `S` includes the cached source prefix.
Require BF16 Q/K/V, equal K/V shapes, matching batch and head dimensions,
and `Hq % Hkv == 0`. For query head `h`, use KV head
`floor(h / (Hq / Hkv))`; copying K/V into `Hq` heads would spend the bandwidth
the grouped layout is meant to save. The model selects the tiled path for a
supported BF16/D128 prefill with `L >= 9`. The direct operator rejects shorter
queries; model attention routes them to the readable fallback.

Start with the supplied causal GQA witness. Its 33 query rows and 47 source
positions cross both tile boundaries, so a result that only handles complete
tiles cannot pass:

```bash
pdm run build-ext
pdm run test --week 2 --day 5 -- -k task_1
```

The direct attention comparison checks BF16 output against the readable
result with `atol=0.02, rtol=0.02`; passing it does not measure speed.

The intended computation is the same scaled dot-product attention as the
readable control. Process **BQ32** query rows and **BK16** source positions at a
time with four 32-lane SIMD groups. Cooperatively load a 32×128 Q tile and a
16×128 K tile, accumulate score fragments, then reuse the K/V storage for the
matching V tile. The source layout uses 128 threads and 12,928 bytes of Q and
K/V threadgroup storage; those sizes describe the implementation, not its
speed. Two padding elements per shared-tile row account for the storage above
the raw Q plus K/V payload. Guard both partial tails before any load or output
write.

For each query row, keep an FP32 running maximum `m`, exponential sum `l`, and
weighted value accumulator `a`. If the next source tile contributes maximum
`m_b`, sum `l_b`, and accumulator `a_b`, merge them as follows:

$$
m' = \max(m,m_b), \qquad
l' = e^{m-m'}l + e^{m_b-m'}l_b,
$$

$$
a' = e^{m-m'}a + e^{m_b-m'}a_b,
\qquad \mathrm{output} = a'/l'.
$$

In words, choose the larger of the old and tile maxima. Multiply the old sum
and accumulator by the exponential of old maximum minus new maximum; multiply
the tile sum and accumulator by the exponential of tile maximum minus new
maximum. Add each pair, then divide the new accumulator by the new sum.
Rescale the old numerator **and** denominator when a later tile raises the
maximum. Padded score cells contribute no probability mass. Return BF16 only
after the final FP32 normalization. This avoids an allocated `L × S` score
workspace, but it still reads dense K/V; a caller-supplied additive mask can
itself have `L × S` entries.

## Task 2: Preserve Masks and the Readable Fallback

Accept no mask, the string `"causal"`, or a broadcastable additive array mask.
Causality is aligned to the end of the cached prefix: query row `i` may see
source positions through `S - L + i`. For `L = 3` and `S = 7`, the first new
query can see positions 0–4. Applying a triangle as if the source length were
three would erase useful cached context. Broadcast an explicit mask over batch
and query-head dimensions, then apply it before the online-softmax update.

A fully masked row has no probability mass. Skip tiles with no valid scores
and make the row's output finite and all zero; do not divide by zero or
propagate `NaN`. The focused mask witness uses
nine BF16 query rows, 17 source positions, and an all-negative-infinity
additive mask:

```bash
pdm run build-ext
pdm run test --week 2 --day 5 -- -k task_2
```

In `Qwen3MultiHeadAttention.__call__`, keep readable grouped attention for
one-token decode, `L <= 8`, and any model shape outside the tiled BF16/D128
boundary. The readable path must receive the same cache update, scale, mask,
query-to-KV-head mapping, and output dtype. Keep separate diagnostic counts
for `tiled_prefill`, `tiled_prefill_fallback`, and `readable`: the fallback count
records that the tiled feature was enabled but ineligible, while `readable`
records the actual computation. These are reference diagnostics to inspect;
the supplied learner tests check public output rather than internal counter
values. A green long-query operator comparison alone does not verify decode.

If you choose the operator off-ramp, keep this model and cache interface and
substitute the equivalent MLX attention operator or readable grouped equation
at this boundary. `--solution mlx` instead runs a separate complete model.

## Task 3: Run `tiled-prefill` in the Model

Wire the eligible attention call through `Qwen3ModelWeek2` and its transformer
layers. The `tiled-prefill` checkpoint retains every Day 4 operator, bounded
cache, and packed projection; it changes only supported dense prefill
attention. Check its focused model behavior and run the live model:

```bash
pdm run build-ext
pdm run test --week 2 --day 5 -- -k task_3
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint tiled-prefill --model qwen3-0.6b --max-tokens 16
```

The supplied three-token model witness guards the readable branch. The
ten-token BF16/D128 model witness is eligible for tiled attention; the 33×47
operator witness also crosses both tile tails. Inspect the reference fallback
and tiled dispatch counts separately. The matched 128-token product workload exercises
the tiled branch in the real model; a matching output alone cannot tell you
which kernel ran.

## Finish the Cumulative `selected` Model

`selected` is the named final Week 2 checkpoint, not an additional Metal
kernel. It keeps the `tiled-prefill` feature set: request-bounded KV capacity,
register-cached RMSNorm, the tiled prefill selector, packed W4 projections,
SIMD matrix prefill, RoPE, and SwiGLU. Each of the capacity, RMSNorm, and tiled
attention controls can also be switched off independently for a matched
counterfactual. The short attention fallback and wider-row RMSNorm fallback
remain part of this model.

First verify the supplied selected-model controls, then run the same completed
course model through the public CLI:

```bash
pdm run test --week 2 --day 5 -- -k selected
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint selected --model qwen3-0.6b --max-tokens 16
```

The test checks the cumulative model and its independently disabled controls;
the model comparison uses `atol=0.75, rtol=0.05`. Its observable assertions
do not prove a particular private dispatch. The CLI run shows that the named
final checkpoint can answer a request. Then run the complete Day 5 learner gate:

```bash
pdm run test --week 2 --day 5
```

Once complete, `selected` is the
default Week 2 checkpoint, but name it explicitly in measurements so the
comparison remains legible.

## Measure the Matched Product

Compare the saved pre-edit `swiglu` row with a new `swiglu` row first. Then
compare `swiglu`, `tiled-prefill`, `selected`, and the full MLX model under one
cached 0.6B request, device, prefill-logit mode, and warmup count:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-swiglu --variant week2-tiled-prefill \
  --variant week2-selected --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day5-product.json
```

A 128-token prompt crosses the `L >= 9` selector boundary. Report the prefill
and complete-request results separately; the `selected` and `tiled-prefill`
rows name the same default feature set, so do not count any difference between
them as a second mechanism gain. Inspect which kernel category dominates under
the same 0.6B model and 128-token prefill shape:

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case swiglu:prefill:128 --case tiled-prefill:prefill:128 \
  --case selected:prefill:128 --warmup 4 --iterations 12 \
  --json-output week2-day5-attribution.json
```

This attribution replays kernel groups on synthetic token IDs; it does not
replace the complete-request comparison. Record whether the tiled path ran
and what later category dominates. An operator result or a historical 4B number
cannot establish a speedup for this 0.6B checkout. The
[performance appendix](./appendix-performance.md) keeps earlier measurements
and unavailable rows as historical evidence. A matched 4B follow-up is
optional after caching that model; repeat every compared row at 4B rather than
mixing model sizes.

Week 2 now supplies a tested single-request model boundary for
[Week 3](./week3-overview.md), where paging and batching add new state and
scheduling work. It does not turn one request into a production serving policy.

{{#include copyright.md}}
