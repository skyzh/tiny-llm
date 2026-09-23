# 🚧 Week 2 Day 5: Tiled Dense Prefill Attention

The readable attention path materializes scores for `L` incoming query rows and
`S` cached source positions. Day 5 asks whether prefill can compute the same
dense attention while keeping only a small online-softmax state.

This is a **prefill** optimization. One-token decode stays on the readable
baseline. The new learner-owned seams are `dense_prefill_attention_mma`, its
private native binding, the BQ32/BK16 Metal kernel, and model selection/counters.

## First Diagnostic: Causal GQA With a Tail

```bash
pdm run build-ext
pdm run test --week 2 --day 5 -- -k causal_gqa
```

The supplied witness uses BF16, head dimension 128, grouped-query attention,
and query/source lengths that cross tile boundaries. An initial failure should
reach the incomplete native tiled path rather than a missing public API.

The wrapper contract is:

```text
Q: B, Hq,  L, D
K: B, Hkv, S, D
V: B, Hkv, S, D
output: B, Hq, L, D
```

Require matching K/V shapes, `Hq % Hkv == 0`, BF16 input, `D = 128`, and
compatible batch/dimensions. Map each query head to its KV head; do not duplicate
K/V merely to make the head counts equal.

## Tile Queries and Source Positions

Use query blocks of **BQ32** rows and source blocks of **BK16** positions. Guard
both tails: `L` or `S` need not be a multiple of the tile size. The accepted
source geometry uses 128-thread groups and 12,928 bytes of Q+K/V threadgroup
storage. These are implementation facts, not occupancy or bandwidth readings.

For each query row, maintain FP32 running maximum `m`, exponential sum `l`, and
weighted value accumulator `a`. When a block produces maximum `m_b`, sum `l_b`,
and accumulator `a_b`, merge it with the previous state:

$$
m' = \max(m,m_b),\qquad
l' = e^{m-m'}l + e^{m_b-m'}l_b,
$$

$$
a' = e^{m-m'}a + e^{m_b-m'}a_b.
$$

After the final source block, return `a / l` in BF16. Rescaling both numerator
and denominator is essential when a later tile raises the maximum.

This schedule does not allocate an `L × S` score workspace. It still reads
dense K/V, and an explicitly supplied additive mask may itself be `L × S`.

## Preserve Mask Semantics

Support:

- no mask;
- the string `"causal"`;
- a broadcastable additive array mask.

Causality is aligned to the end of the cached prefix. With `L = 3` and `S = 7`,
the first incoming query can see source positions 0 through 4, not merely 0.
Apply an explicit mask before the online-softmax update and preserve its
batch/head/query indexing.

A fully masked row has no softmax mass. Return a finite all-zero row rather than
dividing zero by zero or propagating `NaN`:

```bash
pdm run test --week 2 --day 5 -- -k fully_masked
```

## Select the Tiled Path and Keep the Fallback

The current selector admits BF16 D128 prefill with `L >= 9`. Shorter queries,
including decode, use readable grouped attention. Increment:

- `tiled_prefill` when the custom kernel runs;
- `tiled_prefill_fallback` when the feature is enabled but the shape is below
  the tiled boundary;
- `readable` whenever the readable path executes.

The fallback must preserve the same cache update, scale, mask, grouped-head
mapping, output shape, and dtype. The supplied model witness includes a short
query and checks both fallback and readable counters.

## Complete the `tiled-prefill` Checkpoint

```bash
pdm run build-ext
pdm run test --week 2 --day 5
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint tiled-prefill --model qwen3-0.6b --max-tokens 16
```

The whole gate covers a causal GQA tail, a fully masked finite-zero row, the
complete model checkpoint, and the readable fallback. If the optimized shape is
ineligible or fails, retain `swiglu` and readable attention.

## Measure and Decide

Use the incoming cumulative checkpoint as the product control:

```bash
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-0.6b --max-tokens 16
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint tiled-prefill --model qwen3-0.6b --max-tokens 16
```

Confirm the dispatch counter before interpreting time. Accepted component
evidence found tiled prefill **54.07%** faster at 2K and **54.89%** faster at 8K.
Those are operator results. The complete-request effect is smaller and depends
on prompt/output shape and interactions with the other mechanisms.

Keep the path for supported prefill when its numerical, mask, tail, and product
evidence hold. Keep readable attention for short/ineligible queries regardless
of that decision.


## Finish with the selected product

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint selected --model qwen3-0.6b --max-tokens 16
```

The earlier `kv-cache` checkpoint remains the concatenating cache control.
Within the selected path, fixed-width RMSNorm handles dimensions above 4096,
and readable grouped attention handles queries below the tiled boundary. A
fallback is part of the contract, not a silent failure.


The `selected` checkpoint uses request-bounded KV capacity, register-cached
RMSNorm, and tiled dense prefill attention alongside the cumulative packed-W4,
SIMD-matrix, RoPE, and SwiGLU work. It is the named completed model, so the
command above must run the same request through the model rather than only
checking a kernel in isolation. Verify the selected checkpoint's feature set
with the supplied model-free test and preserve the readable attention path for
one-token decode and ineligible shapes:

```bash
pdm run test --week 2 --day 5
```

The [performance appendix](./appendix-performance.md) separates component
measurements from complete-request medians. The accepted selected campaign
reached an 80%-of-MLX direction on 128/128, 512/128, and 2K/16, but missed
it on 2K/128; 2K/512 and 8K/128 product rows are unavailable. Those missing
rows are not implied by the component gains. Week 2 finishes with a tested
single-request path, not a batching or production-serving claim.


{{#include copyright.md}}
