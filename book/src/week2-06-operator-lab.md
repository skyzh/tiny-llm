# 🚧 Week 2 Day 6 (Optional): Long-Context Dense-KV Decode Attention

Day 5 improves packed projections, but the whole-model sweep shows a different
cost growing with context. From a 128-token prompt to 32,640 tokens, the
full-MLX baseline's decode rate falls from 88.52 to 25.71 tokens/s while TPOT
rises from 11.298 to 38.888 ms. In normalized isolated decode replays, QKV and
output projections plus the attention core rise from 36.24% to 62.14%.

This optional chapter tests one bounded response: a memory-efficient
online-softmax kernel for one- or two-row decode against a contiguous dense KV
cache. It does not add paging, chunking, or paged attention. Those mechanisms
belong to Week 3.

The goal is a correct path, a safe fallback, and a matched decision. Do not
assume the candidate passes its performance gate.

## Why the Short Control Is Not Enough

The retired Day 6 experiment dispatched its custom attention path only through
context 256. That made the 128-token result useful for correctness but unable
to test the long-context cost that selected this chapter.

Materializing a full prefill score tensor also stops being viable near the
native endpoint. For 32 query heads and a 32,640-token prompt, a naive FP32
score tensor would occupy approximately

```text
32 * 32,640 * 32,640 * 4 bytes = 127.002 GiB
```

That is a lower-bound argument for memory-efficient attention, not a claim that
the Day 6 decode kernel solves long-prompt prefill. Day 6 keeps only one or two
query rows in flight and walks dense K/V blocks without storing the complete
score row.

## Freeze the Workload

Use Qwen3-4B's model shape:

- 32 query heads, 8 KV heads, and head dimension 128;
- query rows 1 and 2;
- context lengths 128, 512, 2,048, and 8,192;
- BF16 query, key, value, and output with FP32 online-softmax state;
- contiguous dense K/V, the model scale, tails, and causal or no-mask
  semantics; arbitrary explicit masks use the readable fallback.

Generate exactly 128 output tokens for the product comparison. The first token
belongs to prefill/TTFT; calculate TPOT over the remaining 127 decode
intervals. Run the same contexts, order, warmups, and samples for the Day 5
control and candidate.

The canonical selector is `long-context-attention`. First check the learner
seam, then run the Day 5 control and candidate in balanced fresh processes:

```bash
pdm run build-ext
pdm run test --week 2 --day 6

pdm run bench-week2-progression --offline --solution tiny_llm --matrix \
  --variant week2-simd-matmul \
  --variant week2-long-context-attention \
  --prompt-length 128 --prompt-length 512 \
  --prompt-length 2048 --prompt-length 8192 --prompt-length 32640 \
  --output-len 128 --warmup 2 --repeats 4 --prefill-logits last \
  --json-output week2-day6-matrix.json

pdm run bench-week2-progression --offline --solution tiny_llm --matrix \
  --variant week2-simd-matmul \
  --variant week2-long-context-attention \
  --prompt-length 128 --prompt-length 512 \
  --prompt-length 2048 --prompt-length 8192 --prompt-length 32640 \
  --output-len 128 --warmup 2 --repeats 4 --prefill-logits last \
  --disable-week2-long-context-attention \
  --json-output week2-day6-disabled.json
```

The JSON keeps each sample's dispatch counters. A nonzero
`long_context_attention` count proves only that the candidate ran; the disabled
record must return that count to zero.

## Task 1: Preserve Grouped Attention Semantics

Implement online softmax without materializing the full score row:

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

Map every group of four query heads to one KV head. Preserve scale, dense-cache
offsets, tail blocks, and causal or no-mask semantics. Route an arbitrary
explicit mask unchanged through the readable fallback. Accumulate maxima,
denominators, and value-weighted sums in FP32 before returning BF16.

## Task 2: Own the Candidate and Fallback

Wire the smallest complete learner-owned path for
`long-context-attention`. Fill `long_context_attention` and
`should_use_long_context_attention`, then connect them in
`Qwen3MultiHeadAttention.__call__`. Complete
`tiny_llm_ext::long_context_attention`, `Week2DecodeAttention::eval_cpu`, and
`Week2DecodeAttention::eval_gpu` in `src/extensions/src/week2_kernels.cpp`.
Complete the `week2_decode_attention` Metal kernel in
`src/extensions/src/week2_kernels.metal`. The public attention call remains
stable; equivalent private helpers are fine.

The dispatcher must:

- select the candidate only for the declared one- and two-row shapes;
- count candidate selections so the product run proves that it executed;
- expose a disable-only control that always chooses the readable dense path;
- fail safely to that path for unsupported dtype, shape, mask, head mapping,
  layout, or context;
- return the same public result and cache behavior as the control.

Test supported shapes, odd tail lengths, grouped heads, scale, causal/no-mask
behavior, explicit-mask fallback, offsets, and each remaining fallback boundary
before timing. A selection counter proves routing; it does not prove a speedup.

> **Optional profiling evidence.** A 128/8K component replay can corroborate
> the attention crossover, but it does not replace the matched product matrix,
> selection counter, or disable-only control above.

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case simd-matmul:decode:128 \
  --case long-context-attention:decode:128 \
  --case simd-matmul:decode:8192 \
  --case long-context-attention:decode:8192 \
  --warmup 4 --iterations 12 \
  --json-output week2-day6-components.json
```

## Task 3: Run the Matched Decision Gate

Compare Day 5 and `long-context-attention` at the four declared comparison
lengths, then use 32,640 as the native-endpoint safety row. Record the complete
workload identity and, for every pair, TTFT, TPOT, throughput, selection count,
and whether the disable control removes the change.

Keep the candidate only if all of these conditions hold:

1. median TPOT at 8K is at least 5% lower;
2. TPOT moves in the same favorable direction in at least three of the four
   context pairs;
3. neither the 128 nor 512 control regresses by more than 2%;
4. disabling the selector removes the measured gain.

Otherwise record `reject` or `inconclusive` and keep the readable dense fallback.
Correctness is required regardless of the performance decision.

Continue to [Day 7](./week2-07-split-k-prefill.md) from Day 5. The Day 7
checkpoint does not inherit this optional branch.

{{#include copyright.md}}
