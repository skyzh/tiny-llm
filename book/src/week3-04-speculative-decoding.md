# Week 3 Day 4: Speculative Decoding

In this chapter, we will implement **speculative decoding** for the tiny-llm
generation loop.

By this point, the model can already decode with a KV cache. That is correct,
but it still uses the large target model once per output token:

```plain
large model -> token 1
large model -> token 2
large model -> token 3
...
```

Speculative decoding changes the shape of that loop. A smaller draft model
proposes several tokens, and the larger target model verifies them in one
forward pass:

```plain
draft model:  propose token 1, token 2, token 3, token 4
target model: verify those tokens together
```

If the target model agrees with the draft tokens, we can emit several tokens
after one target-model call. If it disagrees, we keep the accepted prefix,
replace the first wrong draft token with the target model's token, and rewind
the cached K/V for the rejected suffix.

This chapter uses a deterministic greedy version of speculative decoding. The
sampled version adds probabilistic acceptance, but the cache and verification
structure are the same.

**Readings**

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)

## The Core Idea

Suppose the accepted sequence currently ends with token `A`.

The draft model predicts four more tokens:

```plain
A -> b1 -> b2 -> b3 -> b4
```

Then the target model runs once on:

```plain
[A, b1, b2, b3, b4]
```

The target model's logits tell us:

```plain
target(A)  = t1
target(b1) = t2
target(b2) = t3
target(b3) = t4
target(b4) = t5
```

To compare predictions with drafted tokens, shift the target predictions one
position:

```plain
[A, t1, t2, t3, t4]
```

Now compare against:

```plain
[A, b1, b2, b3, b4]
```

The first token always matches because it is the already accepted seed token.
After that:

- if `t1 == b1`, then `b1` is accepted,
- if `t2 == b2`, then `b2` is accepted,
- if `t3 != b3`, then `b3` and everything after it is rejected.

When all drafted tokens match, the target model has also produced one extra
token, `t5`. That token becomes the seed for the next speculative round.

## Why Rewind Is Needed

Both models use KV cache. During a speculative round:

1. the draft model writes K/V for the tokens it used while drafting,
2. the target model writes K/V for the full verification sequence,
3. some suffix of those tokens may be rejected.

The cache must represent only the accepted prefix. If the target rejects a draft
token, both caches need to forget the rejected suffix:

```plain
accepted: [A, b1]
rejected: [b2, b3, b4]

target cache: rewind rejected verification tokens
draft cache:  rewind drafted tokens that must be regenerated
```

The exact rewind lengths differ because the draft model has not cached the final
draft token yet. In the reference implementation, the draft model is caught up
by one token only when the whole draft window is accepted.

Paged attention from Week 3 already introduced `rewind(n)` for this reason.
Dense Week 2 caches also need the same lifecycle hook.

## Task 1: Shared Greedy Step

```
src/tiny_llm/generate.py
```

Implement a helper inside `speculative_generate` that calls a Week 2 or Week 3
model and returns greedy next tokens:

```plain
logits = model(tokens[None], offset, kv_cache)
logits = logits[:, -n_tokens:, :]
next_tokens = argmax(log_softmax(logits), axis=-1)
```

For normal one-token decode, `n_tokens` is `1`. For target verification,
`n_tokens` is the size of the verification sequence.

## Task 2: Prefill Both Models

Create one KV cache for the draft model and one for the target model:

```python
draft_kv_cache = draft_model.create_kv_cache()
kv_cache = model.create_kv_cache()
```

Then prefill both models with the same prompt. Each prefill should return:

- the first generated token,
- the offset after the prompt tokens have been written into the cache.

The prompt should be encoded with the tokenizer passed to that model.

## Task 3: Draft Tokens

Implement a small helper that runs the draft model for a fixed number of tokens:

```plain
last accepted token -> draft token 1
draft token 1       -> draft token 2
draft token 2       -> draft token 3
draft token 3       -> draft token 4
```

Use `num_drafts = 4` for this teaching implementation.

The helper should update the draft cache and advance the draft offset by one
for each drafted step.

## Task 4: Verify and Rewind

Build the target verification sequence:

```python
draft_tokens = mx.concat([token, mx.array(draft_tokens)])
```

Run the target model once with `n_tokens = num_drafts + 1`.

Then shift the target predictions so they line up with the drafted tokens:

```plain
target predictions: [t1, t2, t3, t4, t5]
shifted compare:    [token, t1, t2, t3, t4]
draft sequence:     [token, b1, b2, b3, b4]
```

Walk from left to right:

- emit every matching token,
- on the first mismatch, rewind both caches to the accepted prefix,
- set `token` to the target model's replacement token,
- continue the next speculative round from that replacement token.

When all tokens match, emit the whole accepted draft window. Then run the draft
model one more step on the final draft token so the draft cache catches up with
the target cache, and use the target model's extra prediction as the next seed.

## Task 5: Stop and Release Caches

Stop when the token to emit is `tokenizer.eos_token_id`. Do not add EOS to the
detokenizer output.

Always release both KV caches in a `finally` block:

```python
finally:
    _release_kv_cache(draft_kv_cache)
    _release_kv_cache(kv_cache)
```

This keeps paged-cache pages reusable after generation finishes or exits early.

## Verify

Run the fast speculative decoding tests:

```bash
pdm run test --week 3 --day 4
```

The tests use deterministic fake models and fake tokenizers. They check the
algorithmic behavior directly:

- accepting a full draft window,
- carrying the target model's extra token into the next round,
- rejecting a bad draft suffix,
- rewinding draft and target caches by the right lengths,
- stopping on EOS without emitting it,
- releasing both caches.

You can also run the reference solution tests:

```bash
pdm run test-refsol --week 3 --day 4
```

To try the real model path after completing earlier chapters, pass both a target
model and a draft model:

```bash
pdm run main --solution tiny_llm --loader week2 \
  --model qwen3-4b --draft-model qwen3-0.6b
```

{{#include copyright.md}}
