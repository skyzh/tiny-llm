# 🚧 Week 3 Day 7 (Optional): Speculative Decoding

> 🚧 This optional chapter is under review and may change.

Speculative decoding uses a smaller draft model to propose several tokens, then
asks the target model to verify them in one call. Accepted draft tokens reduce
the number of target-model decode steps without changing the target
distribution.

This checkpoint in your solution implements **greedy** speculative decoding:
draft tokens are accepted while they match the target model's greedy tokens.
Extending the same loop to sampling requires the probability-correct acceptance
and residual sampling rules; simple token equality is not enough.

Start with the model-free checkpoint. It uses generated token streams and cache
objects, so it gives useful feedback without downloading either model:

```bash
pdm run test --week 3 --day 7 -- -k "proposal_length or target_only"
```

## Objectives

By the end of this chapter, you should be able to:

- generate a bounded proposal with a smaller draft model;
- verify several proposed positions in one target-model call;
- accept the matching prefix and recover at the first mismatch;
- rewind dense and paged caches without corrupting offsets; and
- decide whether acceptance rate offsets draft and verification overhead.

## Prerequisites

- Complete Week 2 cached generation for both the draft and target models.
- Complete the Week 3 paged cache and page-aware attention path.
- Use compatible tokenizers for the two models. A shared token id must represent
  the same text in both vocabularies. Validate that contract before either model
  runs; mismatched prompt encodings, EOS ids, or vocabularies must fail closed.

This extension comes after paged attention for two concrete reasons. Rejected
draft tokens must release pages and repair the valid tail length, and verifying
several proposed tokens at once is a long-query attention call over the paged
prefix. The paged-cache lifecycle and page-aware long-query operator therefore
form the stable interface on which speculative decoding is built.

## Task 1: Reuse the Cache-Rewind Contract

Day 3 already added `rewind(n)` to the common KV-cache interface. Verify that
prerequisite before building the speculative loop. A dense cache removes the
last `n` logical positions. A paged cache must also return pages that become
unused and shorten the valid prefix of the new tail page. Both implementations
accept zero through the current logical length and reject other values before
changing the cache.

Verify zero-length rewind, a rewind within one page, a rewind across page
boundaries, and a full rewind:

```bash
pdm run test --week 3 --day 3 -- -k rewind
```

## Task 2: Produce a Bounded Draft

Choose a small proposal length such as four. Starting from the last accepted
token, run the draft model one token at a time and retain both the proposed
tokens and the draft-cache offset. Stop early at EOS.

Keep the proposal length configurable. A longer proposal reduces target calls
only when the acceptance rate remains high enough to repay the extra draft work.
Use a default of four and treat zero as an explicit target-only fallback:

```python
speculative_generate(
    draft_model,
    model,
    draft_tokenizer,
    tokenizer,
    prompt,
    proposal_length=4,
    max_tokens=256,
)
```

`max_tokens` bounds newly emitted non-EOS tokens. Zero returns without running a
model or creating a cache, invalid values fail before either model runs, and EOS
may stop generation earlier. Keep the explicit default so existing direct
callers remain source-compatible.

After the target-only fallback and bounded proposal work, rerun the focused
cases:

```bash
pdm run test --week 3 --day 7 -- -k "proposal_length or target_only or budget"
```

## Task 3: Verify in One Target Call

Pass the last accepted token followed by the draft proposal to the target model
in one call. Request logits for every supplied position, then compare the target
greedy tokens with the aligned draft sequence.

Keep prompt, proposal, and verification token arrays in a supported 32-bit
integer dtype. Mixing unsigned and signed 32-bit token arrays can promote a
concatenation to 64-bit indices, which quantized embeddings reject.

The first supplied token is already accepted. Starting at the next position,
find the longest matching prefix. If every draft token matches, keep the target
model's next token so generation can continue without an extra target call.

Use the one-call and full-acceptance cases as the next checkpoint:

```bash
pdm run test --week 3 --day 7 -- -k "verification or full_acceptance"
```

## Task 4: Commit or Rewind

Treat cache offsets as a correctness invariant:

- on full acceptance, advance both caches through the accepted proposal and
  synchronize the draft cache with the target's extra token;
- on a mismatch, emit the target token at that position and rewind every later
  speculative position from both caches;
- after either path, assert that draft offset, target offset, and the logical
  length of every layer cache agree.

Exercise mismatch at the first, middle, and final proposed token. Also test a
fully accepted proposal and EOS inside a proposal. Compare the complete output
with ordinary greedy generation from the target model.

```bash
pdm run test --week 3 --day 7
```

Run the integrated path with a small draft model and a larger target model:

```bash
pdm run main --solution tiny_llm_ref --loader week3 \
  --draft-model qwen3-0.6b --model qwen3-4b --max-tokens 64
```

The draft-model CLI is greedy-only. It rejects temperature, top-p, and top-k
sampling options instead of silently ignoring them. Probability-correct sampled
speculation remains a separate future extension.

## Design the Measurement

The `main` command above is a functional smoke test. It does not emit paired
target-only and speculative timings, so it is not performance evidence.

For a performance decision, run ordinary cached target generation and
speculative generation in balanced fresh processes with the same prompt,
tokenizer, `--max-tokens` value, seed, and synchronization boundary. For
example, compare the command above with the same command without
`--draft-model`, keeping `--max-tokens 64` on both. Verify identical greedy
output, then report proposal length, accepted tokens per proposal, target
verification calls, draft-model time, target-model time, cache maintenance time,
and end-to-end tokens per second for both paths. Record the raw samples and
process order.

Until such a paired artifact exists, this chapter makes no speedup claim.
Acceptance rate alone omits draft work, verification, synchronization, and
cache maintenance.

{{#include copyright.md}}
