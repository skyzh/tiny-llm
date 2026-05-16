# Week 1: From Matmul to Text

In this week, we will start from the basic matrix operations and see how those these matrix manipulations can turn the
Qwen3 model parameters into a model that generates text. We will implement the neural network layers used in the Qwen3
model using mlx's matrix APIs.

We will use `Qwen/Qwen3-0.6B-MLX-4bit` for this week. Week 1 dequantizes model parameters into bfloat16, so start with the 0.6B checkpoint before trying larger Qwen3 models.

In this course, a checkpoint means the saved model package we download from Hugging Face: config, tokenizer files, quantized weights, and the metadata needed by `mlx_lm.load(...)` to rebuild the model. The MLX checkpoint we downloaded in the setup is an int4 quantized version of the original bfloat16 model.

## What We will Cover

* Attention, Multi-Head Attention, and Grouped/Multi Query Attention
* Positional Embeddings and RoPE
* Use `mx.fast.rms_norm` for Qwen3's per-head Q/K normalization in attention, then implement RMSNorm ourselves
* Implement the MLP layer, put the attention layers together, and implement the whole Transformer model
* Load the Qwen3 model parameters and generate text

## What We will Not Cover

To make the journey as interesting as possible, we will skip a few things for now:

* How to quantize/dequantize a model -- that will be part of week 2. The Qwen3 model is quantized so we will need to
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

## Qwen3 Models

You can try the Qwen3 model with MLX/vLLM. You can read the blog post below to have some idea of what we will build
within this course. At the end of this week, we will be able to chat with the model -- that is to say, use Qwen3 to
generate text, as a causal language model.

The reference implementation of the Qwen3 model can be found in huggingface transformers, vLLM, and mlx-lm. You may
utilize these resources to better understand the internals of the model and what we will implement in this week.

**📚 Readings**

- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Huggingface Transformers - Qwen3](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3)
- [vLLM Qwen3](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py)
- [mlx-lm Qwen3](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

{{#include copyright.md}}
