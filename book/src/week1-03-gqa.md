# Week 1 Day 3: Grouped Query Attention (GQA)

On Day 3, we will implement grouped-query attention (GQA). Qwen3 uses GQA to reduce the computational and memory costs
of the key (K) and value (V) projections. In multi-head attention (MHA), every query (Q) head has a corresponding K and
V head. With GQA, groups of Q heads share K and V heads. Multi-query attention (MQA) is the special case in which every
Q head shares a single K/V head pair.


**Readings**

- [GQA paper](https://arxiv.org/abs/2305.13245)
- [Qwen3 layers in mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)
- [PyTorch scaled dot-product attention with `enable_gqa=True`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [`torchtune.modules.MultiHeadAttention`](https://pytorch.org/torchtune/0.3/generated/torchtune.modules.MultiHeadAttention.html)

## Task 1: Implement `scaled_dot_product_attention_grouped`

You will need to modify the following file:

```
src/tiny_llm/attention.py
```

In this task, we will implement grouped scaled dot-product attention, which forms the core of GQA.

Implement `scaled_dot_product_attention_grouped` in `src/tiny_llm/attention.py`. It is similar to standard scaled dot-product
attention, but it supports a number of query heads that is a multiple of the number of key/value heads.

The main process is the same as standard scaled dot-product attention. The difference is that K and V heads are shared
across multiple Q heads. Instead of `H_q` separate K and V heads, there are `H` K and V heads, each shared by
`n_repeats = H_q // H` query heads.

Reshape `query`, `key`, and `value` so that K and V can be broadcast to the query heads in their respective groups during
the matrix multiplications.

- Separate the `H` and `n_repeats` dimensions in `query`.
- Add a dimension of size 1 for `n_repeats` in `key` and `value` so that they broadcast across each group.

Then perform scaled dot-product attention: matrix multiplication, scaling, optional masking, softmax, and a final matrix
multiplication. Broadcasting handles the head sharing without materializing repeated K and V tensors.

Using broadcasting instead of repeating K and V is more efficient because it avoids creating copies of the same data.

Finally, reshape the result to the expected output shape.

```
N.. is zero or more dimensions for batches
H_q is the number of query heads
H is the number of key/value heads (H_q must be divisible by H)
L is the query sequence length
S is the key/value sequence length
D is the head dimension

query: N.. x H_q x L x D
key: N.. x H x S x D
value: N.. x H x S x D
mask: N.. x H_q x L x S
output: N.. x H_q x L x D
```

In addition to grouped heads, this function supports different query and key/value sequence lengths: Q uses length `L`,
while K and V use length `S`.

You can test your implementation by running the following command:

```bash
pdm run test --week 1 --day 3 -- -k task_1
```

## Task 2: Causal Masking

**Readings**

- [Writing an LLM from scratch, part 9 -- causal attention](https://www.gilesthomas.com/2025/03/llm-from-scratch-9-causal-attention)

In this task, we will add causal masking to grouped attention.

Causal masking prevents attention from reading future tokens. When `mask` is set to the string `"causal"`, apply a causal
mask.

The additive causal mask has shape `(L, S)`, where `L` is the query sequence length and `S` is the key/value sequence length.
Allowed positions contain 0, and masked positions contain `-inf`. When `S` is greater than `L`, shift the diagonal by
`S - L` so that the queries correspond to the final `L` positions in the key/value sequence. For example, if `L = 3`
and `S = 5`, the mask is:

```
0   0   0   -inf -inf
0   0   0   0    -inf
0   0   0   0    0
```

Implement `causal_mask` in `src/tiny_llm/attention.py`, then use it in `scaled_dot_product_attention_grouped`. Note that
our shifted diagonal for `L != S` differs from the default behavior of some attention APIs.

You can test your implementation by running the following command:

```bash
pdm run test --week 1 --day 3 -- -k task_2
```

## Task 3: Qwen3 Grouped Query Attention

In this task, we will implement Qwen3's grouped-query attention. Modify the following file:

```
src/tiny_llm/qwen3_week1.py
```

`Qwen3MultiHeadAttention` implements attention for Qwen3. Follow this pseudocode:

```
x: B, L, E
q = linear(x, wq) -> B, L, H_q, D
k = linear(x, wk) -> B, L, H, D
v = linear(x, wv) -> B, L, H, D
q = rms_norm(q, q_norm)
k = rms_norm(k, k_norm)
q = rope(q, offset=slice(0, L))
k = rope(k, offset=slice(0, L))
(transpose as needed)
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, H_q, L, D  # use float32
(transpose as needed)
x = linear(x, wo) -> B, L, E
```

Qwen3 attention has no Q/K/V projection biases, and it applies RMSNorm to each Q and K head before RoPE. We will implement
the reusable `RMSNorm` layer on Day 4, so call `mx.fast.rms_norm` directly for `q_norm` and `k_norm` today. Use
non-traditional RoPE.

You can test your implementation by running the following command:

```bash
pdm run test --week 1 --day 3 -- -k task_3
```

At the end of the day, you should be able to pass all tests of this day:

```bash
pdm run test --week 1 --day 3
```

{{#include copyright.md}}
