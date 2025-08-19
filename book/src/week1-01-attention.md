# Week 1 Day 1: Attention and Multi-Head Attention

In day 1, we will implement the basic attention layer and the multi-head attention layer. Attention layers take a input
sequence and focus on different parts of the sequence when generating the output. Attention layers are the key building
blocks of the Transformer models.

[📚 Reading: Transformer Architecture](https://huggingface.co/learn/llm-course/chapter1/6)

We use the Qwen2 model for text generation. The model is a decoder-only model. The input of the model is a sequence of
token embeddings. The output of the model is the most likely next token ID.

[📚 Reading: LLM Inference, the Decode Phase](https://huggingface.co/learn/llm-course/chapter1/8)

Back to the attention layer. The attention layer takes a query, a key, and a value. In a classic implementation, all
of them are of the same shape: `N.. x L x D`.

`N..` is zero or some number of dimensions for batches. Within each of the batch, `L` is the sequence length and `D` is
the dimension of the embedding for a given head in the sequence.

So, for example, if we have a sequence of 1024 tokens, where each of the token has a 512-dimensional embedding (head_dim),
we will pass a tensor of the shape `N.. x 1024 x 512` to the attention layer.

## Task 1: Implement `scaled_dot_product_attention_simple`

In this task, we will implement the scaled dot product attention function. We assume the input tensors (Q, K, V) have
the same dimensions. In the next few chapters, we will support more variants of attentions that might not have the same
dimensions for all tensors.

```
src/tiny_llm/attention.py
```

**📚 Readings**

* [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
* [PyTorch Scaled Dot Product Attention API](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (assume `enable_gqa=False`, assume dim_k=dim_v=dim_q and H_k=H_v=H_q)
* [MLX Scaled Dot Product Attention API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) (assume dim_k=dim_v=dim_q and H_k=H_v=H_q)
* [Attention is All You Need](https://arxiv.org/abs/1706.03762)

Implement `scaled_dot_product_attention` following the below attention function. The function takes key, value, and query of the same dimensions, and an optional mask matrix `M`.

$$
  \text{Attention} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}} + M)V
$$

Note that $\frac{1}{\sqrt{d_k}}$ is the scale factor. The user might specify their own scale factor or use the default one.

```
L is seq_len, in PyTorch API it's S (source len)
D is head_dim

key: N.. x L x D
value: N.. x L x D
query: N.. x L x D
output: N.. x L x D
scale = 1/sqrt(D) if not specified
```

You may use `softmax` provided by mlx and implement it later in week 2.

Because we are always using the attention layer within the multi-head attention layer, the actual tensor shape when serving
the model will be:

```
key: 1 x H x L x D
value: 1 x H x L x D
query: 1 x H x L x D
output: 1 x H x L x D
mask: 1 x H x L x L
```

.. though the attention layer only cares about the last two dimensions. The test case will test any shape of the batching dimension.

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

Implement `SimpleMultiHeadAttention`. The layer takes a batch of vectors, maps it through the K, V, Q weight matrixes, and use the attention function we implemented in task 1 to compute the result. The output needs to be mapped using the O
weight matrix.

You will also need to implement the `linear` function in `basics.py` first. For `linear`, it takes a tensor of the shape `N.. x I`, a weight matrix of the shape `O x I`, and a bias vector of the shape `O`. The output is of the shape `N.. x O`. `I` is the input dimension and `O` is the output dimension.

For the `SimpleMultiHeadAttention` layer, the input tensors `query`, `key`, `value` have the shape `N x L x E`, where `E` is the dimension of the
embedding for a given token in the sequence. The `K/Q/V` weight matrixes will map the tensor into key, value, and query
separately, where the dimension `E` will be mapped into a dimension of size `H x D`, which means that the token embedding
gets mapped into `H` heads, each with a dimension of `D`. You can directly reshape the tensor to split the `H x D` dimension
into two dimensions of `H` and `D` to get `H` heads for the token.

Now, you have a tensor of the shape `N.. x L x H x D` for each of the key, value, and query. To apply the attention function, you first need to transpose them into shape `N.. x H x L x D`.

* This makes each attention head an independent batch, so that attention can be calculated separately for each head across the sequence `L`.
* If you kept `H` behind `L`, attention calculation would mix head and sequence dimensions, which is not what we want — each head should focus only on the relationships between tokens in its own subspace.

The attention function produces output for each of the head of the token. Then, you can transpose it back into `N.. x L x H x D` and reshape it
so that all heads get merged back together with a shape of `N.. x L x (H x D)`. Map it through the output weight matrix to get
the final output.

```
E is hidden_size or embed_dim or dims or model_dim
H is num_heads
D is head_dim
L is seq_len, in PyTorch API it's S (source len)

w_q/w_k/w_v: E x (H x D)
output/input: N x L x E
w_o: (H x D) x E
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
