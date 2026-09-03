# Week 1 Day 2: Positional Encodings and RoPE

The Day 2 starter already declares `RoPE(dims, seq_len, base=10000, traditional=False)`. Its constructor and call method
are empty. You will fill in those two methods: cache one table of position-dependent angles, then use it to rotate the
last dimension of an input shaped `(N, L, H, D)`. Day 3 will apply the non-traditional form to Qwen3's query and key
heads before attention.

**📚 Readings**

- [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding)
- [Roformer: Enhanced Transformer with Rotary Positional Encoding](https://arxiv.org/pdf/2104.09864)

## Task 1: Implement Traditional Rotary Positional Encoding

You will need to modify the following file:

```text
src/tiny_llm/positional_encoding.py
```

Start by building the frequency table in `RoPE.__init__`. Let `M = D // 2`. Pair index `i`, where `0 <= i < M`, has
the angular rate below; multiplying it by a token position gives the angle for that pair.

```text
angular_rate[i] = base ** (-i / (D // 2))
angle[position, i] = position * angular_rate[i]
```

Use `mlx.core` operations such as `arange`, `power`, `outer`, `cos`, and `sin` to precompute the cosine and sine of those
angles for every position from `0` through `seq_len - 1`. The two tables have shape `(seq_len, M)`. Implement the
operator yourself with these array operations; `mx.fast.rope` is the supplied test's correctness oracle, not the
implementation for this exercise. For this lesson, assume that `D` is even, so `M` pairs cover the whole head
dimension.

If `offset` is not provided, apply positions 0 through `L - 1` to the input sequence. Otherwise, select positions from
the supplied slice. For example, with `offset=slice(5, 10)`, the input sequence must have length 5, and its first token
uses the frequency for position 5. Reshape the selected `(L, M)` basis to broadcast across the batch and head axes.

For Week 1, you only need to support `offset=None` and a single `slice`. We will implement `list[slice]` for continuous
batching later. For now, assume that every item in a batch uses the same offset.

```text
x: (N, L, H, D)
cos/sin_freqs: (MAX_SEQ_LEN, D // 2)
```

Traditional RoPE interprets adjacent values along head dimension `D` as complex-number pairs. If `D = 8`, then `x[0]`
and `x[1]` form one pair, `x[2]` and `x[3]` form another, and so on. Both values in a pair use the same frequency from
`cos_freqs` and `sin_freqs`.

```text
output[0] = x[0] * cos_freqs[0] + x[1] * -sin_freqs[0]
output[1] = x[0] * sin_freqs[0] + x[1] * cos_freqs[0]
output[2] = x[2] * cos_freqs[1] + x[3] * -sin_freqs[1]
output[3] = x[2] * sin_freqs[1] + x[3] * cos_freqs[1]
...and so on
```

You can implement this operation by reshaping `x` to `(N, L, H, D // 2, 2)` and applying the formula to each pair.
Stack the real and imaginary results back along the pair axis, restore the original shape, and return the result in
`x.dtype`.

**📚 Readings**

- [PyTorch RotaryPositionalEmbeddings API](https://pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html)
- [MLX Implementation of RoPE before the custom metal kernel implementation](https://github.com/ml-explore/mlx/pull/676/files)

Run the focused command once before editing. The empty starter returns no array, so the comparison should fail when it
tries to inspect the result. Run the same command again after implementing Task 1; when it passes, your cached basis,
position selection, adjacent pairing, and dtype restoration work together.

```bash
pdm run test --week 1 --day 2 -- -k task_1
```

## Task 2: Implement Non-Traditional `RoPE`

Qwen3 uses a non-traditional arrangement of RoPE pairs. Keep the same cached frequencies and position-selection logic.
When `traditional` is false, split the head dimension into two halves, then pair corresponding values from the halves.
Let `x1 = x[..., :HALF_DIM]` and `x2 = x[..., HALF_DIM:]`.

```text
output[0] = x1[0] * cos_freqs[0] + x2[0] * -sin_freqs[0]
output[HALF_DIM] = x1[0] * sin_freqs[0] + x2[0] * cos_freqs[0]
output[1] = x1[1] * cos_freqs[1] + x2[1] * -sin_freqs[1]
output[HALF_DIM + 1] = x1[1] * sin_freqs[1] + x2[1] * cos_freqs[1]
...and so on
```

Implement this form by selecting the first and second halves of `x` directly, applying the rotations, concatenating the
results, and returning the original dtype. The constructor's default is non-traditional because that is the layout the
Qwen3 attention block will use.

**📚 Readings**

- [vLLM implementation of RoPE](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding)

This focused command should now pass with the half-split layout while reusing the same angles and offset handling:

```bash
pdm run test --week 1 --day 2 -- -k task_2
```

Finally, run both layouts together:

```bash
pdm run test --week 1 --day 2
```

Once that command passes, `RoPE` is ready for Day 3 to rotate Qwen3 query and key heads with one shared slice. Per-request
`list[slice]` offsets remain a later continuous-batching problem.

{{#include copyright.md}}
