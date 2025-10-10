# Week 1: From Matmul to Text

In this week, we will start from the basic matrix operations and see how those these matrix manipulations can turn the
Qwen2 model parameters into a model that generates text. We will implement the neural network layers used in the Qwen2
model using mlx's matrix APIs.

We will use the Qwen2-7B-Instruct model for this week. As we need to dequantize the model parameters, the model of 4GB
download size needs 20GB of memory in week 1. If you do not have enough memory, you can consider using the smaller 0.5B model.

The MLX version of the Qwen2-7B-Instruct model we downloaded in the setup is an int4 quantized version of the original bfloat16 model.

## What We will Cover

* Attention, Multi-Head Attention, and Grouped/Multi Query Attention
* Positional Embeddings and RoPE
* Put the attention layers together and implement the whole Transformer block
* Implement the MLP layer and the whole Transformer model
* Load the Qwen2 model parameters and generate text

## What We will Not Cover

To make the journey as interesting as possible, we will skip a few things for now:

* How to quantize/dequantize a model -- that will be part of week 2. The Qwen2 model is quantized so we will need to
  dequantize them before we can use them in our layer implementations.
* Actually we still used some APIs other than matrix manipulations -- like softmax, exp, log, etc. But they are simple
  and not implementing them would not affect the learning experience.
* Tokenizer -- we will not implement the tokenizer from scratch. We will use the `mlx_lm` tokenizer to tokenize the input.
* Loading the model weights -- I don't think it's an interesting thing to learn how to decode those tensor dump files, so
  we will use the `mlx_lm` to load the model and steal the weights from the loaded model into our layer implementations.

## Basic Matrix APIs

Although MLX does not offer an introductory guide for beginners, its Python API is designed to be highly compatible with NumPy. To get started, you can refer to [NumPy: The Absolute Basic for Beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) to learn essential matrix operations.

You can also refer to the [MLX Operations API](https://ml-explore.github.io/mlx/build/html/python/ops.html#operations)
for more details.

## Qwen2 Models

You can try the Qwen2 model with MLX/vLLM. You can read the blog post below to have some idea of what we will build
within this course. At the end of this week, we will be able to chat with the model -- that is to say, use Qwen2 to
generate text, as a causal language model.

The reference implementation of the Qwen2 model can be found in huggingface transformers, vLLM, and mlx-lm. You may
utilize these resources to better understand the internals of the model and what we will implement in this week.

**📚 Readings**

- [Qwen2.5: A Party of Foundation Models!](https://qwen.ai/blog?id=6da44b4d3b48c53f5719bab9cc18b732a7065647&from=research.research-list)
- [Key Concepts of the Qwen2 Model](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- [Huggingface Transformers - Qwen2](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2)
- [vLLM Qwen2](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2.py)
- [mlx-lm Qwen2](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py)
- [Qwen2 Technical Report](https://arxiv.org/pdf/2407.10671)
- [Qwen2.5 Technical Report](https://arxiv.org/pdf/2412.15115)

{{#include copyright.md}}
