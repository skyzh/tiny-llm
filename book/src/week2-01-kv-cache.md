# Week 2 Day 1: Key-Value Cache

In this chapter, we will implement the **key-value cache** for the Qwen2 model. The key-value cache is an essential component of the attention mechanism, as it allows the model to reuse previously computed results instead of recomputing them for every new token.

**📚 Readings**

- [KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching)

Recall from last week how we supplied data to the model:

```plain
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6]) # returns 7
decode:  _step(model, [1, 2, 3, 4, 5, 6, 7]) # returns 8
decode:  _step(model, [1, 2, 3, 4, 5, 6, 7, 8]) # returns 9
...
```

```plain
x: B, L, E
q = linear(x, wq, bq) -> B, L, H_q, D
k = linear(x, wk, bk) -> B, L, H, D
v = linear(x, wv, bv) -> B, L, H, D
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


Consider two consecutive decoding steps with `L = S = 3` and `L = S = 4`, where each head in each layer has an embedding dim of `D = 4`:

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

Notice that the first three rows/cols of `Q × K^T` are identical in both steps. Also given that we are using the causal masks, we do not need to care about the upper triangle of the matrix. The same applies to the softmax function and the multiplication with the V matrix. This means we are unnecessarily recomputing results for tokens we’ve already processed, and the new information only comes from the last row of `Q * K^T`.

The solution is to cache the K and V matrices and only compute new values for incoming tokens:

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

Each layer in the model maintains its own key-value cache. The cache has a single API, `update_and_fetch`, which:

1. Takes the newly computed `K` and `V` for incoming tokens.
2. Concatenates them with the existing cached matrices.
3. Returns the full cached `K` and `V`.

For week 2 day 1, you only need to handle `key` and `value`. The `mask` and `mask_length` parameters will remain unused.

You may implement this in `kv_cache.py` as `TinyKvFullCache`:

```plain
L' = new tokens length
L  = total tokens length

update_and_fetch(key, value) -> key, value

key:   B, L', H, D
value: B, L', H, D

self.key   = concat_or_initialize(self.key, key, on the L' dimension)
self.value = concat_or_initialize(self.value, value, on the L' dimension)

self.key:   B, L, H, D
self.value: B, L, H, D

return self.key, self.value
```

## Task 2: Use the Key-Value Cache

```
src/tiny_llm/qwen2_week2.py
```

With the cache in place, update your week 1 Qwen2 implementation to support it. Implement the `Qwen2MultiHeadAttention` class in `qwen2_week2.py`.

* Each layer should use its own cache.
* The model must now accept an `offset` argument, which represents the position of the last token processed.
* This value should match the current sequence length in the cache (you can add assertions to check consistency).
* Both the argument and the cache maintain the offset for debugging purposes.

Example computation flow:

```plain
x: B, L', E
q = linear(x, wq, bq) -> B, L', H_q, D
k = linear(x, wk, bk) -> B, L', H, D
v = linear(x, wv, bv) -> B, L', H, D
q = rope(q, offset=slice(offset, offset + L'))
k = rope(k, offset=slice(offset, offset + L'))
(transpose as needed)
k, v = cache.update_and_fetch(k, v) ; k/v: B, L, H, D, q: B, L', H, D
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L', H_q, D  # at float32 precision
(transpose as needed)
x = linear(x, wo) -> B, L', E
```

We use two different variables for the `L'` because they have different meanings in the context of this chapter
and the context of week 1 day 3: in the GQA implementation, k/v's sequence length is `S` (source length), while
q's sequence length is `L`. In the Qwen2 multihead attention implementation, `L'` is the "new token" and `L` is
the total sequence length, which corresponds to `L` and `S` in week 1 respectively.

## Task 3: Implement the Model

```
src/tiny_llm/qwen2_week2.py
```

Complete the rest of the model using your week 1 implementation as a base, but modify all relevant components to use the key-value cache.

To verify correctness, run the following test (almost identical to week 1’s test):

```bash
pdm run test --week 2 --day 1
```

## Task 4: Implement Decoding

```
src/tiny_llm/generate.py
```

Next, implement the decoding logic in `generate.py` by completing the `simple_generate_with_kv_cache` function. This function should call your Week 2 Qwen2 model with both the `offset` and the newly decoded token.

For example:

```plain
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6], 0)  # returns 7
decode:  _step(model, [7], 7)  # returns 8
decode:  _step(model, [8], 8)  # returns 9
...
```

You can test your implementation with:

```bash
pdm run main --solution tiny_llm --loader week2 --model qwen2-0.5b
pdm run main --solution tiny_llm --loader week2 --model qwen2-7b
```

{{#include copyright.md}}
