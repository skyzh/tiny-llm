# Week 1 Day 1: Attention and Multi-Head Attention

On Day 1, we will implement basic attention and multi-head attention. An attention layer processes an input sequence and
weighs the relevance of its different positions when producing each output. Attention is a key building block of Transformer
models.

[📚 Reading: Transformer Architecture](https://huggingface.co/learn/llm-course/chapter1/6)

We use Qwen3, a decoder-only model, for text generation. The model takes a sequence of token IDs, maps them to embeddings,
and produces logits for the next token at each sequence position. The generation loop will later use the final position's
logits to choose the next token ID.

[📚 Reading: LLM Inference, the Decode Phase](https://huggingface.co/learn/llm-course/chapter1/8)

An attention layer takes a query, a key, and a value. In a basic implementation, all three have the same shape:
`N.. x L x D`.

`N..` represents zero or more batch dimensions. Within each batch, `L` is the sequence length and `D` is the embedding
dimension for one attention head.

For example, a sequence of 1,024 tokens with a head dimension of 512 is represented by a tensor of shape
`N.. x 1024 x 512`.

## Task 1: Implement `scaled_dot_product_attention_simple`

In this task, we will implement scaled dot-product attention. We assume that the input tensors Q, K, and V have the same
shape. Later chapters will introduce attention variants whose input shapes differ.

```
src/tiny_llm/attention.py
```

**📚 Readings**

* [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
* [PyTorch Scaled Dot Product Attention API](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (assume `enable_gqa=False`, assume dim_k=dim_v=dim_q and H_k=H_v=H_q)
* [MLX Scaled Dot Product Attention API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) (assume dim_k=dim_v=dim_q and H_k=H_v=H_q)
* [Attention is All You Need](https://arxiv.org/abs/1706.03762)

Implement `scaled_dot_product_attention_simple` using the formula below. The function takes query, key, and value tensors
with the same shape, plus an optional additive mask `M`.

$$
  \text{Attention} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}} + M)V
$$

Here, $\frac{1}{\sqrt{d_k}}$ is the default scale factor. Callers may supply a different scale factor.

```
L is seq_len, in PyTorch API it's S (source len)
D is head_dim

key: N.. x L x D
value: N.. x L x D
query: N.. x L x D
output: N.. x L x D
scale = 1/sqrt(D) if not specified
```

You may use MLX's `softmax`; we will revisit lower-level operations in Week 2.

When this function is called from multi-head attention, the tensors will usually have these shapes:

```
key: 1 x H x L x D
value: 1 x H x L x D
query: 1 x H x L x D
output: 1 x H x L x D
mask: 1 x H x L x L
```

The function itself operates on the last two dimensions and must support any number of leading batch dimensions. The mask
only needs a shape that can broadcast to the attention-score shape.

At the end of this task, you should be able to pass the following tests:

```
pdm run test --week 1 --day 1 -- -k task_1
```

## Task 2: Implement `SimpleMultiHeadAttention`

In this task, we will implement the multi-head attention layer.

```
src/tiny_llm/attention.py
```

**📚 Readings**

* [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
* [PyTorch MultiHeadAttention API](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html) (assume dim_k=dim_v=dim_q and H_k=H_v=H_q)
* [MLX MultiHeadAttention API](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html) (assume dim_k=dim_v=dim_q and H_k=H_v=H_q)
* [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2) helps you better understand what key, value, and query are.

Implement `SimpleMultiHeadAttention`. The layer projects batches of query, key, and value vectors with the Q, K, and V
weight matrices, then passes the projections to the attention function from Task 1. Finally, it applies the output projection
O.

First, implement the `linear` function in `basics.py`. It takes a tensor of shape `N.. x I`, a weight matrix of shape
`O x I`, and an optional bias vector of shape `O`. Its output has shape `N.. x O`, where `I` is the input dimension and
`O` is the output dimension.

For `SimpleMultiHeadAttention`, the input tensors `query`, `key`, and `value` have shape `N x L x E`, where `E` is the
embedding dimension for one token. The Q, K, and V projections each map `E` to `H x D`: `H` heads, each with dimension
`D`. Reshape that final projection dimension into separate `H` and `D` dimensions.

You now have a tensor of shape `N x L x H x D` for each projection. Before applying attention, transpose each one to
`N x H x L x D`.

- This treats each attention head as an independent batch, allowing attention to be calculated separately for each head
  across sequence dimension `L`.
- Leaving `H` after `L` would cause the matrix multiplication to mix the head and sequence dimensions. Each head must attend
  only to token relationships within its own subspace.

The attention function produces one output per head. Transpose the result back to `N x L x H x D`, reshape it to
`N x L x (H x D)`, and apply the output projection.

```
E is hidden_size or embed_dim or dims or model_dim
H is num_heads
D is head_dim
L is seq_len, in PyTorch API it's S (source len)

w_q/w_k/w_v: (H x D) x E
output/input: N x L x E
w_o: E x (H x D)
```

At the end of the task, you should be able to pass the following tests:

```
pdm run test --week 1 --day 1 -- -k task_2
```

You can run all tests for the day with:

```
pdm run test --week 1 --day 1
```

{{#include copyright.md}}
