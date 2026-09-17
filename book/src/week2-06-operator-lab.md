# 🚧 Week 2 Day 6 (Optional): Context-Selected Dense Decode Attention

Day 5 gives prefill and decode different packed-projection schedules. The
whole-model sweep exposes the next boundary: as context grows, dense attention
becomes a larger part of each decode step. The full-MLX control falls from
88.52 tokens/s at a 128-token prompt to 54.39 tokens/s at 8,192 tokens, while
the normalized decode replay attributes 62.14% of its native-endpoint work to
QKV/output projections and the attention core.

You will keep the readable grouped-attention path and add one narrow policy:
use an FP32 online-softmax primitive only for one- or two-row BF16 decode at
8K–32K context. Every other request keeps the readable fallback.

The earlier `long-context-attention` experiment selected the same primitive at
shorter contexts. It helped the 8K decode point but did not move in the
favorable direction in three of four context pairs, so that checkpoint was
rejected. This chapter preserves the useful 8K result by making the measured
crossover part of the dispatch contract. It does not turn the rejected result
into a win at shorter contexts.

## Start from the Product Boundary

This kernel accelerates decode after a dense KV cache already exists. It cannot
make a 32,640-token prefill runnable through the readable attention path. That
prefill would materialize an FP32 score tensor of

```text
32 query heads * 32,640 * 32,640 scores * 4 bytes
    = 136,367,308,800 bytes
    = 127.001953 GiB
```

Record the 32,640+128 Week 2 request as **unavailable**, not as a zero or a
failed timing sample. Week 3 Day 3 introduces pages, Day 4 walks them directly,
and Day 5 owns tiled online-softmax long-prefill attention. Reimplementing that
query/KV tiling here would duplicate the later mechanism.

Begin from the Day 5 checkpoint. The public checkpoint for this optional branch
is `context-selected-attention`.

## Task 1: Preserve the Online-Softmax Invariant

Implement the existing dense-GQA primitive without materializing an `L x S`
score tensor for the selected one- or two-row query:

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

Fill the learner-owned `long_context_attention` wrapper, the existing
`tiny_llm_ext::long_context_attention` operation and
`Week2DecodeAttention::eval_cpu`/`eval_gpu` bodies, and the
`week2_decode_attention` Metal kernel. Equivalent private helpers are fine; do
not add a second native operation.

Hold `m`, `l`, and `o` in FP32, then cast the final result to BF16. Map each
group of four query heads to one KV head. Preserve the model scale, cache
offsets, odd tail blocks, and causal or no-mask semantics.

Run the focused checkpoint after completing the recurrence. Its numerical
cases compare causal and no-mask output with readable grouped attention and
exercise an odd context tail; a correct selected result is BF16 even though
the recurrence is FP32.

```bash
pdm run build-ext
pdm run test --week 2 --day 6
```

## Task 2: Own the Selection Boundary

Implement `should_use_context_selected_attention` and connect the primitive in
`Qwen3MultiHeadAttention.__call__`. The selected path requires all of these
conditions:

- the selector is enabled;
- Q, K, and V are contiguous-compatible BF16 tensors;
- Q has shape `[B, 32, L, 128]`, where `L` is 1 or 2;
- K and V have matching shape `[B, 8, S, 128]`;
- `8192 <= S <= 32768`; and
- the mask is absent or causal.

Use the readable dense fallback for context 8,191 and below, query length 3 and
above, explicit masks, other dtypes or layouts, other head maps, and invalid
tails. Keep candidate and fallback counters independent. Disabling this
checkpoint must make the candidate counter stay at zero while preserving the
same public attention result and cache lifecycle.

Rerun the focused checkpoint at the policy boundary. It covers context
8,191/8,192, query lengths 1/2/3, explicit-mask fallback, incompatible
dtype/head/layout cases, counters, and the disable control. The test command is
feedback about correctness and routing; it is not performance evidence.

## Task 3: Measure the One Shape the Policy Claims

Generate exactly 128 output tokens. Token 1 belongs to prefill/TTFT; calculate
TPOT over the remaining 127 decode intervals. Compare Day 5 with the candidate
in balanced fresh processes at the four runnable prompt lengths:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --matrix \
  --variant week2-simd-matmul \
  --variant week2-context-selected-attention \
  --prompt-length 128 --prompt-length 512 \
  --prompt-length 2048 --prompt-length 8192 \
  --output-len 128 --warmup 2 --repeats 4 --prefill-logits last \
  --json-output week2-day6-matrix.json

pdm run bench-week2-progression --offline --solution tiny_llm --matrix \
  --variant week2-simd-matmul \
  --variant week2-context-selected-attention \
  --prompt-length 128 --prompt-length 512 \
  --prompt-length 2048 --prompt-length 8192 \
  --output-len 128 --warmup 2 --repeats 4 --prefill-logits last \
  --disable-week2-context-selected-attention \
  --json-output week2-day6-disabled.json
```

The matrix records dispatch counters with each sample. Confirm zero candidate
dispatch at 128, 512, and 2K and nonzero eligible dispatch at 8K before reading
the timing. Then apply the fixed gate:

1. candidate dispatch is zero below 8K and nonzero for every eligible 8K
   decode;
2. median 8K TPOT improves by at least 5%;
3. no 128/512/2K TTFT or TPOT median regresses by more than 2%; and
4. disabling the selector removes the 8K gain.

Keep the policy only if all four conditions hold. A correct primitive, a
nonzero counter, or the earlier isolated 8K result does not substitute for the
matched product gate.

Write down `keep`, `reject`, or `inconclusive`, together with the observation
that would reverse the decision. Performance for the integrated checkpoint is
pending until this exact matrix is measured.

Continue to [Day 7](./week2-07-split-k-prefill.md) from Day 5. Day 7 is
independent of this optional branch.

{{#include copyright.md}}
