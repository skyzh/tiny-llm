# Week 1 Day 2: Positional Encodings and RoPE

In day 2, we will implement the positional embedding used in the Qwen2 model: Rotary Positional Encoding. In a transformer
model, we need a way to embed the information of the position of a token into the input of the attention layers. In Qwen2,
positional embedding is applied within the multi head attention layer on the query and key vectors.

**📚 Readings**

- [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding)
- [Roformer: Enhanced Transformer with Rotary Positional Encoding](https://arxiv.org/pdf/2104.09864)

## Task 1: Implement Rotary Positional Encoding "RoPE"

You will need to modify the following file:

```
src/tiny_llm/positional_encoding.py
```

In traditional RoPE (as described in the readings), the positional encoding is applied to each head of the query and key vectors.
You can pre-compute the frequencies when initializing the `RoPE` class.

If `offset` is not provided, the positional encoding will be applied to the entire sequence: 0th frequency applied to the
0th token, up to the (L-1)-th token. Otherwise, the positional encoding will be applied to the sequence according to the
offset slice. If the offset slice is 5..10, then the sequence length provided to the layer would be 5, and the 0th token
will be applied with the 5th frequency.

You *only* need to consider `offset` being `None` or a single slice. The `list[slice]` case will be implemented when we
start implementing the continuous batching feature. Assume all batches provided use the same offset.

```
x: (N, L, H, D)
cos/sin_freqs: (MAX_SEQ_LEN, D // 2)
```

In the traditional form of RoPE, each head on the dimension of `D` is viewed as consecutive complex pairs. That is to
say, if D = 8, then, x[0] and x[1] are a pair, x[2] and x[3] are another pair, and so on. A pair gets the same frequency
from `cos/sin_freqs`.

Note that, practically, D can be even or odd. In the case of D being odd, the last dimension of `x` doesn’t have a matching pair,
and is typically left untouched in most implementations. For simplicity, we just assume that D is always even.

```
output[0] = x[0] * cos_freqs[0] + x[1] * -sin_freqs[0]
output[1] = x[0] * sin_freqs[0] + x[1] * cos_freqs[0]
output[2] = x[2] * cos_freqs[1] + x[3] * -sin_freqs[1]
output[3] = x[2] * sin_freqs[1] + x[3] * cos_freqs[1]
...and so on
```

You can do this by reshaping `x` to (N, L, H, D // 2, 2) and then applying the above formula to each pair.

**📚 Readings**

- [PyTorch RotaryPositionalEmbeddings API](https://pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html)
- [MLX Implementation of RoPE before the custom metal kernel implementation](https://github.com/ml-explore/mlx/pull/676/files)

You can test your implementation by running the following command:

```
pdm run test --week 1 --day 2 -- -k task_1
```

## Task 2: Implement `RoPE` in the non-traditional form

The Qwen2 model uses a non-traditional form of RoPE. In this form, the head embedding dimension is split into two halves,
and the two halves are applied with different frequencies. Let's say `x1 = x[.., :HALF_DIM]` and `x2 = x[.., HALF_DIM:]`.

```
output[0] = x1[0] * cos_freqs[0] + x2[0] * -sin_freqs[0]
output[HALF_DIM] = x1[0] * sin_freqs[0] + x2[0] * cos_freqs[0]
output[1] = x1[1] * cos_freqs[1] + x2[1] * -sin_freqs[1]
output[HALF_DIM + 1] = x1[1] * sin_freqs[1] + x2[1] * cos_freqs[1]
...and so on
```

You can do this by directly getting the first half / second half of the embedding dimension of `x` and applying the
frequencies to each half separately.

**📚 Readings**

- [vLLM implementation of RoPE](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding)

You can test your implementation by running the following command:

```
pdm run test --week 1 --day 2 -- -k task_2
```

At the end of the day, you should be able to pass all tests of this day:

```
pdm run test --week 1 --day 2
```

{{#include copyright.md}}
