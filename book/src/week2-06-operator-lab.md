# Historical Week 2: Bounded Decode-Attention Operator Lab

> **Earlier lesson address.** This page preserves the bounded decode-attention operator lab
> explanation and its original links. The current Week 2 learner route
> ships [Day 1: Cache and Measure](./week2-01-kv-cache.md),
> [Day 2: Keep W4 Packed](./week2-02-quantize-model.md),
> [Day 3: SIMD Matrix Prefill](./week2-03-simd-matrix-prefill.md),
> [Day 4: Fused Model Primitives](./week2-04-fused-model-kernels.md), then
> [Day 5: Tiled Dense Prefill Attention](./week2-05-tiled-prefill-attention.md).
> Later checkpoints, day numbers, tests, and commands below belong to an
> earlier all-days course state; do not use them as gates for this checkout.


Day 5 restores the matrix-shaped projection path selected by the fixed
128-token prefill profile. Day 6 asks a different question: can a secondary
operator earn a place for one named workload?

The supplied worked branch is bounded decode attention. It is useful practice
with online softmax, but the checked fixed workload did not identify attention
as the next dominant category. Treat this chapter as an optional experiment,
not a prerequisite for Day 7 and not evidence of a universal bottleneck.

A successful pass ends with a bounded decision, even when the numbers do not
support keeping the branch.

## Choose the Workload Before the Operator

Write down the model, checkpoint, phase, prompt or context length, warmups,
iterations, and comparison rule before editing code. Start from the Day 5
`simd-matmul` checkpoint and record the same workload for the candidate:

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case simd-matmul:decode:128 --case decode-attention:decode:128 \
  --warmup 4 --iterations 12 \
  --json-output week2-day6-attribution.json
```

The supplied branch uses `decode-attention`. An equivalent experiment on a
different measurement-selected secondary category is valid if it preserves the
public checkpoint and decision-record contract. The course grades observable
behavior and reasoning, not a private file path, exact Metal symbol, device
duration, or schedule choice.

## Task 1: Preserve Bounded Decode-Attention Semantics

Use the supplied branch to make that decision concrete. The readable
grouped-attention path materializes score and probability rows; for one query
row, online softmax can combine the reduction and value-weighted sum without
storing the full score row:

```plain
m = -infinity
l = 0
o = 0

for each key/value block:
    scores = q @ key_block.T * scale
    block_max = max(scores)
    new_m = max(m, block_max)
    alpha = exp(m - new_m)
    probabilities = exp(scores - new_m)
    l = alpha * l + sum(probabilities)
    o = alpha * o + probabilities @ value_block
    m = new_m

return o / l
```

Preserve grouped-query head mapping, dense-cache offsets, BF16 inputs and
outputs, FP32 online-softmax state, scale, and the existing mask adapter. Keep
an exact fallback for shapes outside the tested guard. Do not turn a
short-context experiment into a claim about long-context or paged attention.

## Task 2: Implement and Verify the Branch

Implement the smallest complete branch: replace only the existing fail-closed
Day 6 learner surfaces. Keep the public
attention interface stable so Week 3 can reuse it.

The supplied C++ surface is `tiny_llm_ext::decode_attention`, implemented by
`Week2DecodeAttention::eval_cpu` and `Week2DecodeAttention::eval_gpu` in
`src/extensions/src/week2_kernels.cpp`; its Metal entry is
`week2_decode_attention` in `src/extensions/src/week2_kernels.metal`. The
product calls it from `Qwen3MultiHeadAttention.__call__` through
`decode_attention_custom`. Equivalent internal organization is valid when it
preserves this public behavior and fallback.

Build as soon as the branch is wired; run the focused check before the product
path:

```bash
pdm run build-ext
pdm run test --week 2 --day 6

pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint decode-attention --model qwen3-4b
```

Test supported shapes, grouped heads, offsets, and the exact fallback. A valid
solution may use different helper names and internal organization; it must
produce the same public attention behavior and preserve the fallback.

If you completed the old Week 2 Day 5 attention exercise before the course was
reordered, keep that work. The current SIMD checkpoint is
[Day 3](./week2-03-simd-matrix-prefill.md); this historical optional lab and
its former Day 6 commands are separate from the current learner route.

## Task 3: Re-measure and Decide

A passing branch establishes correctness. The final decision comes from
rerunning the same comparison:

Repeat the frozen workload and compare `simd-matmul` with
`decode-attention`. Record:

- the dominant category before the change;
- the category and product effect you actually observed;
- the exact context range and fallback you tested;
- `keep`, `reject`, or `inconclusive`, plus the next falsifying experiment.

The checked M4 Pro result was equivocal: attributed attention changed from
0.837 ms to 0.831 ms (-0.75%), total attributed time rose 0.97%, and the
separate two-sample product control showed decode rising from 74.34 to 76.50
tokens/s (+2.91%). That supports an `inconclusive` worked example, not a
portable speedup claim. Your decision should follow your matched measurement.

Continue to [Day 7](./week2-07-split-k-prefill.md) from `simd-matmul`. The
`split-k` checkpoint intentionally excludes this optional attention branch.

{{#include copyright.md}}
