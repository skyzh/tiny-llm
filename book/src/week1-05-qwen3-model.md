# Week 1 Day 5: The Qwen3 Model

On Day 5, we will combine the components from the previous chapters into the complete Qwen3 model.

Model-level tests require the corresponding model files. Start with the default 0.6B model; download the larger models
only if you want to test them as well:

```bash
hf download Qwen/Qwen3-0.6B-MLX-4bit
# Optional larger models:
hf download Qwen/Qwen3-1.7B-MLX-4bit
hf download Qwen/Qwen3-4B-MLX-4bit
```

Tests that require an unavailable model will be skipped.

## Task 1: Implement `Qwen3TransformerBlock`

```
src/tiny_llm/qwen3_week1.py
```

**📚 Readings**

- [A Simplified Explanation of the Transformer Block](https://medium.com/@akhileshkapse/a-simplified-explanation-of-the-transformer-block-must-read-blog-for-nlp-enthusiasts-12ef240a62ac)
- [Attention is All You Need](https://arxiv.org/pdf/1706.03762)

Qwen3 uses the following Transformer block structure:

```
  input
/ |
| input_layernorm (RMSNorm)
| |
| Qwen3MultiHeadAttention
\ |
  Add (residual)
/ |
| post_attention_layernorm (RMSNorm)
| |
| MLP
\ |
  Add (residual)
  |
output
```

Run the tests for this task with:

```bash
pdm run test --week 1 --day 5 -- -k task_1
```

## Task 2: Implement `Embedding`

```
src/tiny_llm/embedding.py
```

**📚 Readings**

- [LLM Embeddings Explained: A Visual and Intuitive Guide](https://huggingface.co/spaces/hesamation/primer-llm-embedding)

The embedding layer maps token IDs (integers) to vectors of length `embedding_dim`. In this task, you will implement
that lookup operation.

```
Embedding::__call__
weight: vocab_size x embedding_dim
Input: N.. (tokens)
Output: N.. x embedding_dim (vectors)
```

This can be implemented with array indexing.

When input and output embeddings are tied, Qwen3 also uses the embedding weight as a linear projection from hidden vectors
back to vocabulary logits.

```
Embedding::as_linear
weight: vocab_size x embedding_dim
Input: N.. x embedding_dim
Output: N.. x vocab_size
```

Run the tests for this task with:

```bash
# This task's tests use the 0.6B model and tokenizer.
hf download Qwen/Qwen3-0.6B-MLX-4bit
pdm run test --week 1 --day 5 -- -k task_2
```

## Task 3: Implement `Qwen3ModelWeek1`

Now that we have built all the Qwen3 components, we can implement `Qwen3ModelWeek1`.

```
src/tiny_llm/qwen3_week1.py
```

You will not implement the process of reading model parameters from tensor files. Instead, load the model with `mlx_lm`,
then transfer its parameters into our implementation. The `Qwen3ModelWeek1` constructor therefore accepts an MLX model.

The Qwen3 model has the following layers:

```
input
| (tokens: N..)
Embedding
| (N.. x hidden_size); note that hidden_size == embedding_dim
Qwen3TransformerBlock
| (N.. x hidden_size)
Qwen3TransformerBlock
| (N.. x hidden_size)
...
|
RMSNorm 
| (N.. x hidden_size)
Embedding.as_linear OR linear (lm_head)
| (N.. x vocab_size)
output
```

Read the number of layers, hidden size, head dimension, and other configuration values from `mlx_model.args`, whose type
is defined by [`ModelArgs`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py). The loaded weights
are available through `mlx_model.model`; use the Qwen3 implementation and model metadata to identify the corresponding
layer names.

By this point, you have implemented `RMSNorm`. Replace the temporary Day 3 calls to `mx.fast.rms_norm` with
`RMSNorm(head_dim, q_norm, eps=...)` and `RMSNorm(head_dim, k_norm, eps=...)`. They implement the same formula; the built-in
calls existed only to keep the GQA chapter focused on attention.

Different Qwen3 model variants map hidden vectors back to vocabulary logits in different ways. Some tie the input and
output embeddings and use `Embedding.as_linear`; others have a separate `lm_head` linear layer. Select the strategy with
`mlx_model.args.tie_word_embeddings`: if it is `True`, use `Embedding.as_linear`; otherwise, load and use `lm_head`.

The model takes a sequence of token IDs and returns unnormalized logits for every sequence position. On Day 6, we will
use the final position's logits to select the next token and generate a response.

The MLX models used in this course have quantized weights. Dequantize each linear or embedding layer before loading it
into tiny-llm by using the provided `quantize.dequantize_linear` function.

Pass `mask="causal"` to every Transformer block. For a one-token sequence the mask has no effect; for longer sequences,
it prevents each position from attending to future tokens.

Run the tests for this task with:

```bash
# Download each model you want to test. Missing models are skipped.
hf download Qwen/Qwen3-0.6B-MLX-4bit
hf download Qwen/Qwen3-1.7B-MLX-4bit
hf download Qwen/Qwen3-4B-MLX-4bit
pdm run test --week 1 --day 5 -- -k task_3
```

At the end of the day, you should be able to pass all tests of this day:

```bash
pdm run test --week 1 --day 5
```

{{#include copyright.md}}
