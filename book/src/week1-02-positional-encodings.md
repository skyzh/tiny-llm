# Week 1 Day 2: Positional Encodings and RoPE

On Day 2, we will implement the positional encoding used by Qwen3: rotary positional encoding (RoPE). A Transformer needs
a way to represent each token's position in the sequence. Qwen3 applies RoPE to the query and key vectors within its
multi-head attention layer.

**📚 Readings**

- [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding)
- [Roformer: Enhanced Transformer with Rotary Positional Encoding](https://arxiv.org/pdf/2104.09864)

## Task 1: Implement Traditional Rotary Positional Encoding

You will need to modify the following file:

```
src/tiny_llm/positional_encoding.py
```

In traditional RoPE, as described in the readings, positional encoding is applied independently to each head of the query
and key vectors. You can precompute the frequencies when initializing the `RoPE` class.

If `offset` is not provided, apply positions 0 through `L - 1` to the input sequence. Otherwise, select positions from
the supplied slice. For example, with `offset=slice(5, 10)`, the input sequence must have length 5, and its first token
uses the frequency for position 5.

For Week 1, you only need to support `offset=None` and a single `slice`. We will implement `list[slice]` for continuous
batching later. For now, assume that every item in a batch uses the same offset.

```
x: (N, L, H, D)
cos/sin_freqs: (MAX_SEQ_LEN, D // 2)
```

Traditional RoPE interprets adjacent values along head dimension `D` as complex-number pairs. If `D = 8`, then `x[0]`
and `x[1]` form one pair, `x[2]` and `x[3]` form another, and so on. Both values in a pair use the same frequency from
`cos_freqs` and `sin_freqs`.

In practice, `D` can be even or odd. If it is odd, the final value has no partner and is typically left unchanged. For
simplicity, this implementation requires `D` to be even.

```
output[0] = x[0] * cos_freqs[0] + x[1] * -sin_freqs[0]
output[1] = x[0] * sin_freqs[0] + x[1] * cos_freqs[0]
output[2] = x[2] * cos_freqs[1] + x[3] * -sin_freqs[1]
output[3] = x[2] * sin_freqs[1] + x[3] * cos_freqs[1]
...and so on
```

You can implement this operation by reshaping `x` to `(N, L, H, D // 2, 2)` and applying the formula to each pair.

**📚 Readings**

- [PyTorch RotaryPositionalEmbeddings API](https://pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html)
- [MLX Implementation of RoPE before the custom metal kernel implementation](https://github.com/ml-explore/mlx/pull/676/files)

You can test your implementation by running the following command:

```
pdm run test --week 1 --day 2 -- -k task_1
```

## Task 2: Implement Non-Traditional `RoPE`

Qwen3 uses a non-traditional arrangement of RoPE pairs. Split the head dimension into two halves, then pair corresponding
values from the halves. Let `x1 = x[..., :HALF_DIM]` and `x2 = x[..., HALF_DIM:]`.

```
output[0] = x1[0] * cos_freqs[0] + x2[0] * -sin_freqs[0]
output[HALF_DIM] = x1[0] * sin_freqs[0] + x2[0] * cos_freqs[0]
output[1] = x1[1] * cos_freqs[1] + x2[1] * -sin_freqs[1]
output[HALF_DIM + 1] = x1[1] * sin_freqs[1] + x2[1] * cos_freqs[1]
...and so on
```

Implement this form by selecting the first and second halves of `x` directly, applying the rotations, and concatenating
the results.

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
