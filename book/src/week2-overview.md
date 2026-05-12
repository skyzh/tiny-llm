# Week 2: Tiny vLLM

In Week 2 of the course, we will focus on building serving infrastructure for the Qwen3 model. Essentially, this means creating a minimal version of the vLLM project from scratch. By the end of the week, you’ll be able to serve the Qwen3 model efficiently on your Apple Silicon device using the infrastructure we’ve built together.

## What We’ll Cover

* Key-value cache implementation
* C++/Metal kernels
    * Implementing a quantized matmul kernel
    * Implementing a flash attention kernel
    * Note: This week, we won’t focus on performance optimization. The kernels you build will likely be around 10x slower than MLX implementations. Optimizing them will be left as an exercise.
* Model serving infrastructure
    * Implementing chunked prefill
    * Implementing continuous batching

Additionally, the repo includes skeleton code for the Qwen3 model. If your device supports the bfloat16 data type (note: M1 chips do not), you’re encouraged to try implementing it and experiment with the Qwen3-series models as well.

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
