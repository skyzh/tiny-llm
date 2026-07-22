# Week 2: A Step Closer to vLLM

> **Course status:** The new benchmarking, fused-operator, decode-attention,
> and end-to-end optimization chapters are works in progress. The dense KV
> cache and quantized-matmul chapters come from the original course.

Week 2 keeps the readable Week 1 model intact and builds a separate optimized
Qwen3 path for single-request decoding. We begin with measurements, replace the
largest bottlenecks behind explicit interfaces, and finish with an end-to-end
performance target against MLX on the same machine.

## What We Will Cover

- Synchronized benchmarking and a matched MLX baseline
- Vanilla, SIMD-group matrix, and SIMD matrix-vector quantized multiplication
- Fast RMSNorm, RoPE, and SwiGLU operations
- A dense per-request key-value cache for incremental decoding
- The optimized decode-attention primitive
- Last-token-only logits and end-to-end graph cleanup
- An acceptance target of 80-90% of MLX decode throughput

Week 2 does **not** call MLX-provided implementations of the operators we are
learning. The required path implements quantized matmul, RMSNorm, RoPE,
SwiGLU, and decode attention in course-owned Python, C++, or Metal code. In
particular, it does not use `mx.quantized_matmul`, `mx.dequantize`, or the
corresponding `mx.fast` operators as shortcuts.

We are still building on MLX as infrastructure. `mlx_lm` loads the official
Qwen3 4-bit checkpoint and tokenizer. `mlx.core` supplies arrays, lazy graph
evaluation, memory management, device streams, and synchronization. The MLX
extension API registers our C++ primitive and dispatches our Metal kernels.
Those facilities are the platform on which the course implementation runs;
they are not substitutes for the operator implementations themselves.

The order is intentional. Days 2-3 establish quantized model weights, Day 4
adds the fast operators required by the Week 2 model, and Day 5 introduces its
dense KV cache. Decode attention can then consume a real cache on Day 6. A
later chapter never becomes an undeclared prerequisite for an earlier one.

Unlike Week 1, the completed Week 2 model prefills a dense KV cache once,
passes only the new token during decode, keeps its linear and embedding weights
quantized, and imports optimized operations from `week2_kernels.py`. Week 1
continues to use its readable full-prefix generation loop and Python RMSNorm,
RoPE, attention, and MLP implementations.

Week 3 imports these Week 2 interfaces rather than copying or replacing them.
That boundary lets each week's model remain understandable and runnable on its
own.

The percentage estimates at the end of each chapter are planning ranges, not
independent promises. They refer to end-to-end decode throughput on the Week 2
benchmark, and they do not add linearly because the bottleneck changes after
each optimization.

## Expected Performance Contribution

**Estimated overall improvement: 600-750% over the readable Week 1 decode
path, with a goal of at least 80% of matched MLX throughput.** On an M1 Pro,
the current work-in-progress reference measures about 247-252 decode tok/s
versus about 317 tok/s for MLX at a 128-token prompt: roughly 78-79% of MLX and
about 8.1x the original 30.4 tok/s Week 1 baseline. Prefill improved from about
1,005 to 2,052 tok/s when the vanilla quantized matmul was replaced with the
course's `simdgroup_matrix` kernel. The custom decode path is now close to, but
has not yet reached, the 80% acceptance boundary; the synchronized benchmark,
not the estimate, is the authority.

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
