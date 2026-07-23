# Week 2 Day 1: KV Cache

In this chapter, we will add a **key-value cache** to the Qwen3 model. During
generation, the cache lets each attention layer reuse the keys and values from
previous tokens instead of recomputing the entire prefix at every step.

This is the foundation of Week 2 decode optimization, not a serving-only Week 3
feature. Without it, every generated token reruns all model layers over an
ever-growing prefix, overwhelming the gains from faster individual kernels.

**📚 Readings**

- [KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching)

Recall how Week 1 repeatedly supplied the full sequence to the model:

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
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D  # at float32 precision
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

The leading `3 x 3` block of `QK^T` is identical in both steps. A causal mask
also prevents earlier queries from attending to the new token, so their outputs
do not change. Recomputing those rows, their softmax values, and their products
with `V` is wasted work. Only the new query row contributes a new output.

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

Each Transformer layer maintains its own key-value cache. The cache exposes one
method, `update_and_fetch`, which:

1. Accepts the newly computed `K` and `V` for the incoming tokens.
2. Appends them along the sequence dimension.
3. Returns the complete cached `K` and `V`, the updated offset, and the mask.

In this chapter, the cache passes `mask` through unchanged and does not use
`mask_length`. Those parameters become important in Week 3 for batching.

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

## Task 2: Preserve the Week 1 Boundary

```
src/tiny_llm/qwen3_week2.py
```

Keep the readable Week 1 model and its full-prefix generation loop unchanged.
Start a separate `qwen3_week2.py` model with the same dense weights and readable
RMSNorm, RoPE, SwiGLU, and attention equations. Change only the state flow in
this chapter: the Week 2 model accepts a cache and an offset while Week 1 keeps
recomputing the full prefix. This produces the baseline that every later Week 2
chapter will optimize.

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
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, H_q, L, D  # float32
transpose and reshape x to B, L, H_q * D
x = linear(x, wo) -> B, L, E
```

Here, `L` is the number of incoming query tokens and `S` is the total cached
sequence length after the update. This matches the Week 1 GQA convention: `L`
is the query length, while `S` is the key/value source length. During
single-token decoding, `L = 1` and `S` grows by one on each call.

The linear layers, RMSNorm, RoPE, SwiGLU, and attention remain the readable
implementations at this checkpoint. Do not introduce packed weights or fast
kernels yet: measuring one algorithmic change makes the gain attributable.

## Task 3: Create Request-Scoped Caches

```
src/tiny_llm/qwen3_week2.py
```

Implement `create_kv_cache` so every request gets one cache handle per
Transformer layer. Pass the matching layer cache through every block and keep
the caller's offset consistent with the cache's logical length.

To verify correctness, run the following test, which is similar to the Week 1
model test:

```bash
pdm run test --week 2 --day 1
```

## Task 4: Connect the Serving Loop

```
src/tiny_llm/generate.py
```

The first model call prefills the cache with the complete prompt. Each later
call passes only the token produced by the preceding step, together with the
number of tokens already cached. The same lifecycle will be owned by the
continuous-batching scheduler in Week 3.

For example:

```plain
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6], 0)  # returns 7
decode:  _step(model, [7], 6)  # returns 8
decode:  _step(model, [8], 7)  # returns 9
...
```

You can test your implementation with:

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-4b
```

You can also run the same loop with the reference solution:

```bash
pdm run main --solution tiny_llm_ref --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b
```

## Integrate and Measure

Run the cached readable checkpoint end to end before changing any operator:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

Record this number in your optimization ledger. The next chapter teaches how
to compare it fairly with Week 1 and MLX; every later command changes exactly
one cumulative checkpoint.

## Expected Performance Contribution

**Measured cumulative improvement: 19.51 to 101.80 tok/s, or 5.22x over Week 1,
after a 128-token prompt on Qwen3-0.6B on an M4 Pro.** This three-process median
compares the readable full-prefix Week 1 model with the dense cached checkpoint;
no operator kernel has changed yet. The exact ratio depends strongly on context
and output length. Unlike a constant-factor kernel optimization, KV caching
changes each decode step from recomputing the full prefix to processing one new
query token, so the advantage grows throughout a generation.

{{#include copyright.md}}
