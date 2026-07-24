# 🚧 Week 2: A Step Closer to vLLM

> 🚧 This overview and all Week 2 chapters are under review and may change.

Week 2 keeps the readable Week 1 model intact and builds a separate optimized
Qwen3 path for single-request decoding. It begins by changing the algorithm:
prefill once, retain a dense KV cache, and decode one new token at a time. Only
then does a matrix-vector kernel describe the workload we are optimizing.

Every later chapter starts from the runnable checkpoint produced by the
previous chapter. The working loop is always:

```plain
measure and profile -> name the largest relevant cost -> optimize one thing
                    -> verify -> benchmark -> profile again
```

The benchmark decides whether a change stays. The follow-up profile explains
what became expensive next. There is no final chapter where a pile of isolated
operators suddenly becomes a model, and a kernel is never introduced merely
because it sounds useful.

Week 2 inherits Week 1's BF16 model-storage contract. Dense and quantized
weights, activations, projections, KV-cache entries, and model-facing kernel
outputs are BF16. Numerically sensitive reductions, dot products, and
online-softmax state accumulate in FP32 inside readable expressions or kernel
registers. Week 2 extensions are GPU-only: readable Python/MLX equations and
vanilla Metal kernels provide correctness references without requiring CPU BF16
support. This contract remains in force for Week 3, so later chapters only
describe new storage and scheduling behavior.

## What We Will Cover

- A dense per-request key-value cache for incremental decoding
- Synchronized benchmarking and Metal profiling of the cached baseline
- A readable quantized matrix product and a SIMD matrix-vector decode kernel
- The course-owned decode-attention primitive
- Fast RMSNorm, RoPE, and SwiGLU operations
- A BF16 SIMD-matrix quantized prefill kernel
- A measured split-K schedule for small Qwen prefill matrices
- A last-token output interface for generation
- Acceptance targets of 80% of MLX prefill and decode throughput on the
  fixed Week 2 checkpoint

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
Qwen3-4B 4-bit checkpoint and tokenizer. `mlx.core` supplies arrays, lazy graph
evaluation, memory management, device streams, and synchronization. The MLX
extension API registers our C++ primitive and dispatches our Metal kernels.
Those facilities are the platform on which the course implementation runs;
they are not substitutes for the operator implementations themselves.

The order is intentional:

1. **KV cache:** copy the readable Week 1 operators into a Week 2 model, add
   request-scoped state, and stop recomputing the prefix.
2. **Benchmark and profile:** measure the cached model against MLX, then rank
   real GPU costs rather than guessing what should be slow.
3. **Quantized matvec:** the decode profile points at projection weight reads,
   so keep weights packed and integrate the SIMD matrix-vector kernel. Reprofile
   to expose the next context-dependent cost.
4. **Decode attention:** the context sweep shows attention growing, so replace
   the readable score/softmax/value composition with online softmax and measure
   its retained range.
5. **Fast kernels:** after the large decode kernels shrink, the profile exposes
   repeated RMSNorm, RoPE, and SwiGLU launches. Fuse them one at a time and
   remeasure after each checkpoint.
6. **SIMD-matrix prefill:** switch to the prefill profile, where quantized
   matrix multiplication dominates. Introduce 8×8 matrix fragments with FP32
   accumulation and benchmark real Qwen projection shapes.
7. **Split-K prefill:** the Day 6 shape sweep reveals under-filled grids only
   for small Qwen K/V projections. Partition their reduction dimension, merge
   BF16 partial storage with an FP32 final sum, and fall back to Day 6 at the
   measured crossover.

A later chapter never becomes an undeclared prerequisite for an earlier one.

## Why FlashAttention Waits Until Week 3

Week 2 intentionally does not add dense FlashAttention. The matched prefill
profile identifies quantized projection matmul as the dominant cost, and the
Day 6 kernel already brings the completed checkpoint close to MLX at the fixed
acceptance shape. A second dense-attention schedule would not follow the
measured bottleneck and would be replaced as soon as Week 3 makes paged K/V the
canonical serving layout.

Instead, Week 2 teaches the ingredients that remain useful: Day 4 introduces
online softmax and Day 6 introduces cooperative SIMD-matrix tiling. Week 3 adds
page-table translation and combines all three ideas in one paged
FlashAttention operator. A dense first-prefill fast path is a reasonable
follow-up experiment when the cache is empty, but it is not a required Week 2
implementation or a second model-facing attention interface.

Unlike Week 1, the completed Week 2 model prefills a dense KV cache once,
passes only the new token during decode, keeps its linear and embedding weights
quantized, dispatches separate decode and prefill matrix schedules, and imports optimized operations from
`week2_kernels.py`. Week 1 continues to use its readable full-prefix generation
loop and Python RMSNorm, RoPE, attention, and MLP implementations.

Week 3 imports these Week 2 interfaces rather than copying or replacing them.
Its paged-attention chapters combine Day 4 online softmax and Day 6 matrix
fragments only after page-table translation has been introduced, while the
quantized projections inherit Day 7's measured dispatch. That boundary
lets each week's model remain understandable and runnable on its own.

The cumulative ladder is executable at any time. The
[performance appendix](./appendix-performance.md) records the matched results:

```bash
pdm run bench-week2-progression --offline --repeats 3 \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last
```

The runner executes each checkpoint in a fresh process and reports its median
against Week 1 and MLX. It also records the MLX version because that baseline
changes. The performance appendix records the cumulative percentages in one
place. They are not additive promises: replacing one bottleneck changes how
much every later replacement matters.

The acceptance shape uses a 128-token prompt followed by 128 timed decode
steps; `--output-len 129` includes the first token produced by prefill.
It is long enough to amortize compilation and launch noise while remaining in
Week 2's single-request interactive scope. Publish 2K and longer context sweeps
as diagnostics too; they expose the dense-attention boundary that motivates
Week 3, but do not silently replace the fixed acceptance denominator.

The default runs the reference checkpoints. After implementing the cumulative
selector in your model, add `--solution tiny_llm` to measure your own complete
ladder. Preserve the named checkpoints as you work; a later implementation
should add a new branch without changing what an earlier checkpoint executes.

{{#include copyright.md}}

<!--
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
