# Week 1: From Matmul to Text

This week, we will start with basic array and matrix operations and use them to turn Qwen3 model parameters into a model
that generates text. We will implement the neural network layers used by Qwen3 with MLX's array APIs.

## Where You Start

After [setup](./setup.md), you have run the reference model and watched it
generate text. Your own package, `src/tiny_llm`, has the same files and
function signatures, but the functions you will write are stubs whose bodies
are just `pass`. Each day, you fill in a few of them, copy that day's tests
with `pdm run test --week 1 --day N`, and work until they pass.

Days 1 to 4 build the layers one at a time. Day 5 assembles them into the
Qwen3 model and loads the real weights. Day 6 adds the generation loop, and
`pdm run main --solution tiny_llm --loader week1` then generates text with
your code. Day 7 adds sampling and prepares the toolchain for Week 2.
[Day 1](./week1-01-attention.md) starts with attention.

## Model and Precision

We will use `Qwen/Qwen3-0.6B-MLX-4bit`. Your model runs on the GPU and stores
its weights and activations in BF16, a 16-bit floating-point format. Start with
the 0.6B model; larger Qwen3 models need more memory. Some tests use small
inputs in other dtypes, because they are easier to check by hand; that does not
change the dtype your model uses.

A few operations, such as attention and RMSNorm, lose accuracy in BF16. For
those, your code converts to FP32, does the arithmetic, and converts the result
back to BF16. Week 1 keeps this simple and readable, even though it creates a
full FP32 copy of the tensor. In Week 2 you write kernels that keep the tensors
in BF16 and use FP32 only inside the kernel's registers.

## What We Will Cover

- Attention, multi-head attention, grouped-query attention, and multi-query attention
- Positional encodings and RoPE
- Using `mx.fast.rms_norm` for Qwen3's per-head Q/K normalization, then implementing RMSNorm ourselves
- Implementing the MLP, combining the attention components, and building the complete Transformer model
- Loading Qwen3 model parameters and generating text

## What We Will Not Cover

To make the journey as interesting as possible, we will skip a few things for now:

- Quantization and dequantization internals. These will be covered in Week 2. For now, we use a provided helper to
  dequantize the Qwen3 weights before passing them to our layer implementations.
- Low-level implementations of operations such as softmax, exponentiation, and logarithms. These operations are simple
  enough that using the MLX versions does not detract from the learning objectives.
- Tokenization. We use the `mlx_lm` tokenizer rather than implementing one from scratch.
- Decoding model-weight files. We use `mlx_lm` to load the model, then transfer its weights into our layer implementations.

## Basic Matrix APIs

MLX's Python API is designed to be familiar to NumPy users. If you are new to array programming, start with
[NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html).

You can also refer to the [MLX Operations API](https://ml-explore.github.io/mlx/build/html/python/ops.html#operations)
for more details.

## Qwen3 Models

You can run Qwen3 with MLX or vLLM. The readings below provide context for what we will build. By the end of the week,
you will be able to use Qwen3 as a causal language model to generate text.

Reference implementations of Qwen3 are available in Hugging Face Transformers, vLLM, and mlx-lm. Use them to explore
the model's internals and compare them with this week's implementation.

**📚 Readings**

- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Hugging Face Transformers — Qwen3](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3)
- [vLLM Qwen3](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py)
- [mlx-lm Qwen3](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

{{#include copyright.md}}
