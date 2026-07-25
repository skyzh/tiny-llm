# 🚧 Week 3 Optional Extension: Speculative Decoding

> 🚧 This optional chapter is under review and may change.

Speculative decoding uses a smaller draft model to propose several tokens, then
asks the target model to verify them in one call. Accepted draft tokens reduce
the number of target-model decode steps without changing the target
distribution.

This checkpoint in your solution implements **greedy** speculative decoding:
draft tokens are accepted while they match the target model's greedy tokens.
Extending the same loop to sampling requires the probability-correct acceptance
and residual sampling rules; simple token equality is not enough.

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
  the same text in both vocabularies.

This extension comes after paged attention for two concrete reasons. Rejected
draft tokens must release pages and repair the valid tail length, and verifying
several proposed tokens at once is a long-query attention call over the paged
prefix. The paged-cache lifecycle and page-aware long-query operator therefore
form the stable interface on which speculative decoding is built.

## Task 1: Make Cache Rewind a Contract

Add `rewind(n)` to the common KV-cache interface. A dense cache removes the last
`n` logical positions. A paged cache must also return pages that become unused
and shorten the valid prefix of the new tail page.

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

## Task 3: Verify in One Target Call

Pass the last accepted token followed by the draft proposal to the target model
in one call. Request logits for every supplied position, then compare the target
greedy tokens with the aligned draft sequence.

The first supplied token is already accepted. Starting at the next position,
find the longest matching prefix. If every draft token matches, keep the target
model's next token so generation can continue without an extra target call.

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

Run the integrated path with a small draft model and a larger target model:

```bash
pdm run main --solution tiny_llm_ref --loader week3 \
  --draft-model qwen3-0.6b --model qwen3-4b
```

## Measure the Decision

Report proposal length, accepted tokens per proposal, target verification
calls, draft-model time, target-model time, and end-to-end tokens per second.
Compare against ordinary cached target generation on the same prompt. Keep
speculative decoding optional when draft quality, cache rewind, or verification
overhead makes it slower.

Do not infer a speedup from acceptance rate alone: a successful serving result
must include both models, verification, synchronization, and cache maintenance.

{{#include copyright.md}}
