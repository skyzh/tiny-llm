# Week 2: A Step Closer to vLLM

> **Course status:** The new benchmarking, fused-operator, decode-attention,
> and end-to-end optimization chapters are works in progress. The quantized
> matmul chapter comes from the original course.

Week 2 keeps the readable Week 1 model intact and builds a separate optimized
Qwen3 path for single-request decoding. We begin with measurements, replace the
largest bottlenecks behind explicit interfaces, and finish with an end-to-end
performance target against MLX on the same machine.

## What We Will Cover

- Synchronized benchmarking and a matched MLX baseline
- Quantized matrix-vector and matrix-matrix multiplication
- Fast RMSNorm, RoPE, and SwiGLU operations
- The optimized decode-attention primitive
- Last-token-only logits and end-to-end graph cleanup
- An acceptance target of 80-90% of MLX decode throughput

The required path uses MLX's production quantized matmul because the course's
small custom SIMD QMV is educational, not competitive with MLX across machines.
The custom kernel remains a measured stretch exercise with direct correctness
tests at one and eight input rows.

We will continue using the official Qwen3 MLX 4-bit checkpoints. Unlike Week 1,
the Week 2 model keeps its linear and embedding weights quantized and imports
optimized operations from `week2_kernels.py`. Week 1 continues to use its
readable Python RMSNorm, RoPE, attention, and MLP implementations.

Week 3 imports these Week 2 interfaces rather than copying or replacing them.
That boundary lets each week's model remain understandable and runnable on its
own.

{{#include copyright.md}}

<!--
https://github.com/ml-explore/mlx/blob/main/mlx/backend/cpu/quantized.cpp
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py
MLX uses INT4 W4A16
https://ml-explore.github.io/mlx/build/html/dev/extensions.html
https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal
https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h#L962

pdm run ./build_ext.sh

speculative decoding
prefill and decode separation
quantized kv cache
Assert return data type

https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h
https://github.com/philipturner/metal-flash-attention
https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h

attention mask why
https://www.shashankshekhar.com/blog/apple-metal-vs-nvidia-cuda
https://arxiv.org/pdf/2308.16369

padding
https://huggingface.co/docs/transformers/pad_truncation

https://siboehm.com/articles/22/CUDA-MMM
https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal

pdm run batch-main --solution ref --model qwen3-4b --prefill-step 16
-->
