# 🚧 Week 2: A Step Closer to vLLM

> 🚧 This overview was substantially revised and is a work in progress.

> **Course status:** All Week 2 chapters are under review after this restructure.
> The dense KV-cache and quantized-matmul topics come from the original course;
> the benchmarking, decode-attention, and fused-kernel checkpoints are new.

Week 2 keeps the readable Week 1 model intact and builds a separate optimized
Qwen3 path for single-request decoding. It begins by changing the algorithm:
prefill once, retain a dense KV cache, and decode one new token at a time. Only
then does a matrix-vector kernel describe the workload we are optimizing.

Every later chapter starts from the runnable checkpoint produced by the
previous chapter. Implement one replacement, integrate it into the Week 2
model immediately, verify correctness, and measure the new end-to-end decode
rate before continuing. There is no final chapter where a pile of isolated
operators suddenly becomes a model.

## What We Will Cover

- A dense per-request key-value cache for incremental decoding
- Synchronized benchmarking of the cached baseline against MLX
- A readable quantized matrix product and a SIMD matrix-vector decode kernel
- The course-owned decode-attention primitive
- Fast RMSNorm, RoPE, and SwiGLU operations
- A last-token output interface for generation
- An acceptance target of 70% of MLX decode throughput

Week 2 does **not** call MLX-provided implementations of the operators we are
learning. The required path implements quantized matmul, decode attention,
RMSNorm, RoPE, and SwiGLU in course-owned Python, C++, or Metal code. In
particular, the completed checkpoint does not use `mx.quantized_matmul`,
`mx.dequantize`, `mx.fast` operators, or
`mx.fast.scaled_dot_product_attention` as shortcuts. The Day 1 baseline still
uses Week 1's provided `mx.dequantize` loading helper to materialize readable
dense weights; Day 3 replaces that loading path as part of keeping weights
packed.

We are still building on MLX as infrastructure. `mlx_lm` loads the official
Qwen3 4-bit checkpoint and tokenizer. `mlx.core` supplies arrays, lazy graph
evaluation, memory management, device streams, and synchronization. The MLX
extension API registers our C++ primitive and dispatches our Metal kernels.
Those facilities are the platform on which the course implementation runs;
they are not substitutes for the operator implementations themselves.

The order is intentional:

1. **KV cache:** copy the readable Week 1 operators into a Week 2 model, add
   request-scoped state, and stop recomputing the prefix.
2. **Benchmark:** measure that cached model and the matched cached MLX baseline.
3. **Quantized matvec:** keep weights packed, integrate the SIMD decode kernel,
   and measure the first operator replacement.
4. **Decode attention:** replace the exact Week 1 float32 attention composition
   with the course-owned online-softmax kernel and measure the whole model.
5. **Fast kernels:** replace RMSNorm, then RoPE, then SwiGLU. Each replacement
   is integrated and benchmarked before the next one begins.

A later chapter never becomes an undeclared prerequisite for an earlier one.

Unlike Week 1, the completed Week 2 model prefills a dense KV cache once,
passes only the new token during decode, keeps its linear and embedding weights
quantized, and imports optimized operations from `week2_kernels.py`. Week 1
continues to use its readable full-prefix generation loop and Python RMSNorm,
RoPE, attention, and MLP implementations.

Week 3 imports these Week 2 interfaces rather than copying or replacing them.
That boundary lets each week's model remain understandable and runnable on its
own.

The cumulative ladder is executable at any time:

```bash
pdm run bench-week2-progression --offline --repeats 3 \
  --model qwen3-0.6b --input-len 128 --output-len 65 --warmup 2
```

The runner executes each checkpoint in a fresh process and reports its median
against Week 1 and MLX. The percentages at the end of each chapter are
cumulative measurements, not additive promises: replacing one bottleneck
changes how much every later replacement matters.

The default runs the reference checkpoints. After implementing the cumulative
selector in your model, add `--solution tiny_llm` to measure your own complete
ladder. Preserve the named checkpoints as you work; a later implementation
should add a new branch without changing what an earlier checkpoint executes.

## Expected Performance Contribution

**Measured overall improvement: about 12.4x over the readable Week 1 decode
path, with a goal of at least 70% of matched MLX throughput.** On an M4 Pro,
the minimal path retains the dense KV cache, quantized SIMD matvec, decode
attention, RMSNorm, RoPE, and SwiGLU. The three-process cumulative run measured
242.67 decode tok/s versus 327.84 tok/s for MLX: 74.0%.
Prefill tiling, direct quantized embedding, detailed last-token analysis, and
further scheduling experiments move to the Week 3 performance lab. The small
shared `logits_to_keep` interface remains available to Week 2 generation. The
synchronized benchmark, not the estimate, is the authority.

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
