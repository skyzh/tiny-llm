# Week 1 Day 5: The Qwen2 Model

In day 5, we will implement the Qwen2 model.

Before we start, please make sure you have downloaded the models:

```bash
huggingface-cli download Qwen/Qwen2-0.5B-Instruct-MLX
huggingface-cli download Qwen/Qwen2-7B-Instruct-MLX
```

Otherwise, some of the tests will be skipped.

## Task 1: Implement `Qwen2TransformerBlock`

```
src/tiny_llm/qwen2_week1.py
```

**📚 Readings**

- [A Simplified Explanation of the Transformer Block](https://medium.com/@akhileshkapse/a-simplified-explanation-of-the-transformer-block-must-read-blog-for-nlp-enthusiasts-12ef240a62ac)
- [Attention is All You Need](https://arxiv.org/pdf/1706.03762)

Qwen2 uses the following transformer block structure:

```
  input
/ |
| input_layernorm (RMSNorm)
| |
| Qwen2MultiHeadAttention
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

You should pass all tests for this task by running:

```bash
# Download the models if you haven't done so
huggingface-cli download Qwen/Qwen2-0.5B-Instruct-MLX
huggingface-cli download Qwen/Qwen2-7B-Instruct-MLX
# Run the tests
pdm run test --week 1 --day 5 -- -k task_1
```

## Task 2: Implement `Embedding`

```
src/tiny_llm/embedding.py
```

**📚 Readings**

- [LLM Embeddings Explained: A Visual and Intuitive Guide](https://huggingface.co/spaces/hesamation/primer-llm-embedding)

The embedding layer maps one or more tokens (represented as an integer) to one or more vector of dimension `embedding_dim`.
In this task, you will implement the embedding layer.

```
Embedding::__call__
weight: vocab_size x embedding_dim
Input: N.. (tokens)
Output: N.. x embedding_dim (vectors)
```

This can be done with a simple array index lookup operation.

In the Qwen2 model, the embedding layer can also be used as a linear layer to map the embeddings back to the token space.

```
Embedding::as_linear
weight: vocab_size x embedding_dim
Input: N.. x embedding_dim
Output: N.. x vocab_size
```

You should pass all tests for this task by running:

```bash
# Download the models if you haven't done so; we need to tokenizers
huggingface-cli download Qwen/Qwen2-0.5B-Instruct-MLX
huggingface-cli download Qwen/Qwen2-7B-Instruct-MLX
# Run the tests
pdm run test --week 1 --day 5 -- -k task_2
```

## Task 3: Implement `Qwen2ModelWeek1`

Now that we have built all the components of the Qwen2 model, we can implement the Qwen2ModelWeek1 class.

```
src/tiny_llm/qwen2_week1.py
```

**📚 Readings**

- [Qwen2.5-7B-Instruct model parameters](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct?show_file_info=model.safetensors.index.json)

In this course, you will not implement the process of loading the model parameters from the tensor files. Instead, we
will load the model using the `mlx-lm` library, and then we will place the loaded parameters into our model. Therefore,
the `Qwen2ModelWeek1` class will take a MLX model as the constructor argument.

The Qwen2 model has the following layers:

```
input
| (tokens: N..)
Embedding
| (N.. x hidden_size); note that hidden_size==embedding_dim
Qwen2TransformerBlock
| (N.. x hidden_size)
Qwen2TransformerBlock
| (N.. x hidden_size)
...
|
RMSNorm 
| (N.. x hidden_size)
Embedding::as_linear  OR  Linear (lm_head)
| (N.. x vocab_size)
output
```

You can access the number of layers, hidden size, and other model parameters from `mlx_model.args`. Note that different
size of the Qwen2 models use different strategies to map the embeddings back to the token space. For the 0.5b model, it
directly uses the `Embedding::as_linear` layer. For the 7b model, it has a separate `lm_head` linear layer. You can
decide which strategy to use based on the `mlx_model.args.tie_word_embeddings` argument. If it is true, then you should
use `Embedding::as_linear`. Otherwise, the `lm_head` linear layer will be available and you should load its parameters.

The input to the model is a sequence of tokens. The output is the logits (probability distribution) of the next token.
In the next day, we will implement the process of generating the response from the model, and decide the next token
based on the probability distribution output.

Also note that the MLX model we are using (Qwen2-7B/0.5B-Instruct) is a quantized model. Therefore, you also need to
dequantize the weights before loading them into our tiny-llm model. You can use the provided `quantize::dequantize_linear`
function to dequantize the weights.

You also need to make sure that you set `mask=causal` when the input sequence is longer than 1. We will explain why
in the next day.

You should pass all tests for this task by running:

```bash
# Download the models if you haven't done so
huggingface-cli download Qwen/Qwen2-0.5B-Instruct-MLX
huggingface-cli download Qwen/Qwen2-7B-Instruct-MLX
# Run the tests
pdm run test --week 1 --day 5 -- -k task_3
```

At the end of the day, you should be able to pass all tests of this day:

```bash
pdm run test --week 1 --day 5
```

{{#include copyright.md}}
