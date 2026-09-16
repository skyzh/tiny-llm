# 🚧 Week 2 Day 1: KV Cache

Your Week 1 Qwen model already generates by rerunning the full prefix. Day 1
keeps that path intact while you complete four separate Week 2 shells:

- `src/tiny_llm/kv_cache.py::TinyKvFullCache` stores one layer's dense K/V;
- `src/tiny_llm/qwen3_week2.py::Qwen3ModelWeek2` threads cache state and
  offsets through the model;
- `Qwen3ModelWeek2.create_kv_cache` creates one cache per layer and request;
- `src/tiny_llm/generate.py` prefills once, then sends only the new token.

Together, these pieces make prefill populate the cache and make decode send
only the new token. The starter already supplies the Week 1 operators and the
model-loading boundary. Start with the focused learner gate:

```bash
pdm run test --week 2 --day 1
```

When it passes, run the `kv-cache` checkpoint shown in Task 4. That live call
puts the cache into the generation loop instead of exercising it only as an
isolated data structure.

Each attention layer can then reuse the keys and values from previous tokens
instead of recomputing the entire prefix at every step.

This is the foundation of Week 2 decode optimization. Week 3 will change how
the cache is stored and shared, but the reuse starts here. Without it, every
generated token reruns all model layers over an ever-growing prefix and can
overwhelm gains from faster individual kernels.

**📚 Readings**

- [KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching)

First, make the repeated work concrete. Week 1 supplied the full sequence to
the model on every step:

```plain
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6]) # returns 7
decode:  _step(model, [1, 2, 3, 4, 5, 6, 7]) # returns 8
decode:  _step(model, [1, 2, 3, 4, 5, 6, 7, 8]) # returns 9
...
```

```plain
x: B, L, E
q = linear(x, wq) -> B, L, H_q, D
k = linear(x, wk) -> B, L, H, D
v = linear(x, wv) -> B, L, H, D
q = rms_norm(q, q_norm)
k = rms_norm(k, k_norm)
q = rope(q, offset=slice(offset, offset + L))
k = rope(k, offset=slice(offset, offset + L))
(transpose as needed)
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D
# q/k/v and the returned model tensor are BF16; the Python `mlx.core` expression may use FP32 intermediates
(transpose as needed)
x = linear(x, wo) -> B, L, E
```

The attention mechanism is computed as:

$$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$


Consider two consecutive decoding steps with `L = S = 3` and `L = S = 4`.
Assume that each attention head has dimension `D = 4`:

```
L = 3
Q        x  K^T     =         
1 1 1 1     1 2 3      1x1  -inf -inf
2 2 2 2     1 2 3      2x1  2x2  -inf
3 3 3 3     1 2 3      3x1  3x2  3x3
            1 2 3

L = 4
Q        x  K^T       =
1 1 1 1     1 2 3 4      1x1  -inf -inf -inf
2 2 2 2     1 2 3 4      2x1  2x2  -inf -inf
3 3 3 3     1 2 3 4      3x1  3x2  3x3  -inf
4 4 4 4     1 2 3 4      4x1  4x2  4x3  4x4
```

The leading `3 x 3` block of `QK^T` is identical in both steps. The causal mask
prevents earlier queries from attending to the new token, so those outputs do
not change either. Only the new query row can produce a new output; recomputing
the earlier rows, softmax values, and products with `V` is wasted work.

Instead, cache the previous keys and values and compute only the projections for
incoming tokens:

```
K in cache:
1 1 1 1
2 2 2 2

[a b c d] represent cached values

L = 1, S = 3
Q        x  K^T       =         
            (⬇️ is K not transposed)
            [1 1 1 1]      
            [2 2 2 2]      
3 3 3 3      3 3 3 3      3x1 3x2 3x3

L = 1, S = 4
Q        x  K^T       = 
            (⬇️ is K not transposed)
            [1 1 1 1]      
            [2 2 2 2]      
            [3 3 3 3]
4 4 4 4      4 4 4 4      4x1 4x2 4x3 4x4
```

## Task 1: Implement the Key-Value Cache

```
src/tiny_llm/kv_cache.py
```

Each Transformer layer owns a key-value cache. Its `update_and_fetch` method:

1. Accepts the newly computed `K` and `V` for the incoming tokens.
2. Appends them along the sequence dimension.
3. Returns the complete cached `K` and `V`, the updated offset, and the mask.

For now, pass `mask` through unchanged and leave `mask_length` unused. Week 3
will use both when requests share a batch.

You may implement this in `kv_cache.py` as `TinyKvFullCache`:

```plain
L_new = number of incoming tokens

update_and_fetch(key, value, mask_length, mask) -> key, value, offset, mask

key:   B, H, L_new, D
value: B, H, L_new, D

if self.key_values is None:
    self.key_values = (key, value)
else:
    cached_key, cached_value = self.key_values
    self.key_values = (
        concat(cached_key, key, axis=2),
        concat(cached_value, value, axis=2),
    )

self.offset += L_new
key, value = self.key_values  # B, H, offset, D

return key, value, self.offset, mask
```

Keep this first cache deliberately simple and dense. Each `mx.concat` allocates
a larger buffer and copies the previous K/V contents. Across a token-by-token
decode of length `S`, those copies add up to `O(S²)` bytes even though the cache
avoids `O(S²)` prefix recomputation. The reference cache records that traffic
as `growth_copy_bytes` so the profiler can separate it from attention. Week 3
replaces repeated concatenation with preallocated pages for serving.

## Task 2: Build the Cached Week 2 Model

```
src/tiny_llm/qwen3_week2.py
```

Keep the Week 1 Python model and its full-prefix generation loop unchanged.
Build the separate `qwen3_week2.py` model with the same dense weights and the
Week 1 `mlx.core` RMSNorm, RoPE, SwiGLU, and attention equations. Change only
the state flow: the Week 2 model accepts a cache and an offset, while Week 1
continues to recompute the full prefix. Every later Week 2 chapter starts from
this baseline.

- Give each layer its own cache.
- Add an `offset` argument to the model. It is the number of tokens already in
  the cache, and therefore the position of the first incoming token.
- The argument should match the cache's current sequence length. Assertions can
  make this invariant explicit.
- The caller and cache both track the offset to make consistency checks easier.

Example computation flow:

```plain
x: B, L, E
q = linear(x, wq) -> B, L, H_q, D
k = linear(x, wk) -> B, L, H, D
v = linear(x, wv) -> B, L, H, D
q = rms_norm(q, q_norm)
k = rms_norm(k, k_norm)
q = rope(q, offset=slice(offset, offset + L))
k = rope(k, offset=slice(offset, offset + L))
transpose q, k, v to B, H, L, D
k, v = cache.update_and_fetch(k, v)  # k/v: B, H, S, D; q: B, H_q, L, D
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, H_q, L, D
# q/k/v and the returned model tensor are BF16; attention arithmetic is still the Week 1 `mlx.core` path
transpose and reshape x to B, L, H_q * D
x = linear(x, wo) -> B, L, E
```

Here, `L` is the number of incoming query tokens and `S` is the total cached
sequence length after the update. This matches the Week 1 GQA convention: `L`
is the query length, while `S` is the key/value source length. During
single-token decoding, `L = 1` and `S` grows by one on each call.

The linear layers, RMSNorm, RoPE, SwiGLU, and attention remain the Week 1
Python implementations at this checkpoint. Save packed weights and fast
kernels for later checkpoints so this measurement isolates one algorithmic
change. The model still uses BF16 storage; "Week 1 Python" describes the
implementation style, not a return to an FP32 model.

## Task 3: Create Request-Scoped Caches

```
src/tiny_llm/qwen3_week2.py
```

Implement `create_kv_cache` so each request receives one cache handle per
Transformer layer. Pass the matching cache through each block, and keep the
caller's offset equal to the cache's logical length.

The Day 1 test checks this request-scoped lifecycle together with the cache and
model work from the earlier tasks.

## Task 4: Connect the Serving Loop

```
src/tiny_llm/generate.py
```

Send the complete prompt on the first model call to prefill the cache. On each
later call, send only the token produced by the preceding step and the number
of tokens already cached. Week 3 moves this same lifecycle into the
continuous-batching scheduler.

For example:

```plain
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6], 0)  # returns 7
decode:  _step(model, [7], 6)  # returns 8
decode:  _step(model, [8], 7)  # returns 9
...
```

You can test your solution with:

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-4b
```

You can also run the same loop with the reference solution:

```bash
pdm run main --solution tiny_llm_ref --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-4b
```

## Integrate and Measure

Finish Day 1 with a matched Week 1 versus cached Week 2 observation. The runner
uses fresh processes, applies the same Qwen3-4B 128×129 workload to both rows,
and writes the configuration beside the result:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week1 --variant week2-kv-cache \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --json-output week2-day1-cache.json
```

Keep this JSON as Day 2's baseline. Its useful result is the matched observation
and recorded workload identity, not a speedup claim for another model, prompt
length, output length, or device.

Day 1 changes the generation algorithm by removing full-prefix recomputation,
so measure it with the end-to-end benchmark rather than inventing a
shader-level limiter from a GPU trace. On Day 2, attribute this exact cached
workload and turn the observation into a falsifiable next change. Begin Day 3
only after that evidence names dense projections.

{{#include copyright.md}}
