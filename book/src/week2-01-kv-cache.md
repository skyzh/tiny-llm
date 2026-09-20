# 🚧 KV Cache: Reuse the Prefix

You have a working Week 1 model and a saved measurement of full-prefix
generation. For a six-token prompt, that loop calls the model with six tokens,
then seven, then eight. The earlier tokens pass through the same layers again.
A causal model can retain their keys and values and process only the incoming
token on each decode step:

```text
prefill: model([1, 2, 3, 4, 5, 6], offset=0) → token 7
decode:  model([7], offset=6)                 → token 8
decode:  model([8], offset=7)                 → token 9
```

Build the separate Week 2 model while keeping the Week 1 loop available as your
control. Start with the cache tests:

```bash
pdm run test --week 2 --day 1 -- -k full_cache
```

Expect failures at the empty `TinyKvFullCache` methods initially. Your first
checkpoint is a cache that preserves logical values and appends without copying
the old prefix when capacity is available. Model integration comes afterward.

## Separate Length from Capacity

In `src/tiny_llm/kv_cache.py`, complete `TinyKvFullCache.update_and_fetch`,
`materialize`, and `rewind`. The starter supplies `offset`, `capacity`,
`_key_storage`, `_value_storage`, `key_values`, and `growth_copy_bytes`.

For tensors shaped `B, H, L_new, D`, append on the sequence axis. `offset` is the
number of valid tokens, while `capacity` is the number the backing arrays can
hold. Return only the logical prefix, together with its updated length and the
unchanged mask. Unused capacity must never become visible to attention.

A small example distinguishes append from growth:

| Operation | Logical length | Capacity | What happens to the old prefix? |
|---|---:|---:|---|
| Append three tokens to an empty cache | 3 | 4 | No old prefix |
| Append one token | 4 | 4 | Stays in the existing storage |
| Append one more token | 5 | 8 | Copied once into larger storage |

Reserve the next power-of-two capacity large enough for the required length.
Within that capacity, write the new slice and advance the logical length. On
growth, copy the valid prefix once, append the new values, and expose the new
logical view. Count the old K/V bytes copied during growth in
`growth_copy_bytes`. Repeated concatenation produces correct values but copies
the prefix on every append, so it does not satisfy this storage contract.

The supplied tests check evaluated K/V storage as well as values. A Python
object retaining its name is not proof that its underlying array buffer was
reused. `materialize` evaluates the owned storage without changing its layout.
`rewind(n)` removes the newest `n` logical tokens and rejects invalid lengths;
a rewind to zero makes the logical cache empty without requiring capacity to be
released. The next append must overwrite the abandoned suffix, not expose it.
Keep this method's existing interface for later consumers.

## Connect One Cache per Layer and Request

Implement the shells in `src/tiny_llm/qwen3_week2.py`. Start from the Week 1
attention, MLP, and transformer equations, using their readable RMSNorm, RoPE,
and SwiGLU implementations. The `kv-cache` feature set enables neither packed
weights nor the later custom kernels.

`Qwen3ModelWeek2.create_kv_cache` returns a separate cache for each layer of a
request. In each attention call:

1. Project only the incoming hidden states into Q, K, and V.
2. Normalize Q/K and apply RoPE using the incoming token positions.
3. Append the new K/V to that layer's cache.
4. Attend with the incoming Q against all valid cached K/V.
5. Apply the output projection and continue through the block.

Here `L` is the incoming query length and `S` is the total cached length after
append. Prefill starts with many query rows; one-token decode has `L = 1` while
`S` grows. The model's offset must equal each cache's prior logical length. A
mismatched offset rotates the new token at the wrong position, even if the
stored K/V shapes look plausible; reject it.

Pass the mask through the cache unchanged. Attention owns its interpretation.
Keep the existing `mask_length` argument for compatibility rather than adding
batching behavior to this request-scoped exercise.

In `src/tiny_llm/generate.py`, prefill the prompt once, then send only the newly
generated token with the correct offset. Reuse that request's layer caches
throughout generation. New requests need new logical cache state.

## Complete the Cached Product

Once the cache, model, and generation loop are connected, run the full gate and
the public checkpoint:

```bash
pdm run test --week 2 --day 1
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b
```

The gate checks cache values, capacity behavior, model composition, and offset
agreement. The prompt run exercises the generation path. Neither a cache-only
pass nor plausible generated text substitutes for the other checks.

Now compare with your incoming Week 1 model:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week1 --variant week2-kv-cache \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-cache.json
```

This comparison measures the combined cached-generation checkpoint. Use the
focused storage witness to establish buffer reuse; the complete-request timing
alone cannot attribute a gain specifically to the allocation policy. Record
what changed, then use the cached attribution command from
[Measurement](./week2-02-benchmark-profile.md#find-where-the-cached-model-spends-time).
Your next implementation keeps [projection weights packed](./week2-03-quantize-model.md).

{{#include copyright.md}}
