# 🚧 Week 2 Day 1: Reuse the Prefix

Week 1 sends the whole growing sequence through every layer after each sampled
token. Day 1 changes that loop: prefill the prompt once, retain each layer's K/V,
then project only the new token during decode.

```text
prefill: model([1, 2, 3, 4], offset=0) → token 5
decode:  model([5], offset=4)           → token 6
decode:  model([6], offset=5)           → token 7
```

Your learner-owned seam spans three files:

- `src/tiny_llm/kv_cache.py`: dense append in `TinyKvFullCache`;
- `src/tiny_llm/qwen3_week2.py`: one cache per layer and offset checks;
- `src/tiny_llm/generate.py`: one prefill followed by one-token decode calls.

The readable Week 1 operators remain the fallback and correctness control.

## First Diagnostic: Append One Chunk

Run the narrowest cache witness first:

```bash
pdm run test --week 2 --day 1 -- -k full_cache
```

An initial failure at `update_and_fetch` identifies the missing learner seam.
For tensors shaped `B, H, L_new, D`, append on sequence axis 2, advance
`offset` by `L_new`, pass the mask through unchanged, and return all logical K/V.

The Day 1 implementation is intentionally readable:

```text
if cache is empty:
    cached_k, cached_v = new_k, new_v
else:
    cached_k = concat(cached_k, new_k, axis=2)
    cached_v = concat(cached_v, new_v, axis=2)
offset += L_new
```

This removes repeated model computation, but each concatenation copies the old
logical prefix. Preserve that behavior today: it becomes Day 2's causal control.

## Connect Cache State to Positions

Each Transformer layer needs its own cache for one request. In attention:

1. project the incoming hidden states to Q, K, and V;
2. normalize Q/K and apply RoPE at the incoming `offset`;
3. transpose to `B, H, L, D` and append K/V to the layer cache;
4. attend incoming Q over the complete cached K/V;
5. apply the output projection.

Here `L` is the incoming query length and `S` is the cache's logical length
after append. During decode, `L = 1` while `S` grows. Q has query heads; K/V may
have fewer heads under grouped-query attention.

The caller's offset and every layer cache's old logical length must agree. A
mismatch applies RoPE at the wrong position even when all shapes look valid, so
reject it before mutating the request.

Create new per-layer caches for every new request. Reuse them across that
request's prefill and decode calls. Do not place request state on the model.

## Complete the `kv-cache` Checkpoint

After the focused cache test, run the whole gate and live product path:

```bash
pdm run test --week 2 --day 1
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b --max-tokens 16
```

The whole gate also checks that the checkpoint stays readable and that a model
position disagreeing with its cache is rejected. The model command confirms
that your cache is reached through generation; a data-structure-only pass is
not a completed checkpoint.

If the extension or later custom kernels are unavailable, `kv-cache` remains
the fallback product path because it uses the readable operators.

## Measure and Decide

Run the same prompt and output length for the Week 1 control and `kv-cache`.
The public CLI does not expose a Week 1 checkpoint inside the Week 2 loader, so
record the two commands separately and state that process startup is included:

```bash
/usr/bin/time -p pdm run main --solution tiny_llm --loader week1 \
  --model qwen3-0.6b --max-tokens 16
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b --max-tokens 16
```

Your record should answer four questions:

- Did the cache and complete generation gates pass?
- Did decode send only the new token and keep offsets aligned?
- What happened to coarse complete-request time on the identical request?
- What evidence would make you revisit the change?

Keep the cache when it preserves output behavior and removes full-prefix model
recomputation. Do not claim it eliminated cache-copy traffic: Day 2 measures and
removes that separate cost.

Continue to [request-bounded capacity](./week2-02-benchmark-profile.md).

{{#include copyright.md}}
