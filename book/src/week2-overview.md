# 🚧 Week 2: A Step Closer to vLLM

> **Status: Experimental.** Use the verification matrix below to distinguish
> continuously tested behavior from locally measured performance and work that
> still needs maintainer review.

Week 2 keeps the readable Week 1 model intact and builds a separate optimized
Qwen3 path for single-request decoding. It begins by changing the algorithm:
prefill once, retain a dense KV cache, and decode one new token at a time. Only
then does a matrix-vector kernel describe the workload we are optimizing.

Throughout this week, **MLX** means the framework or its production operators,
the **reference solution** means the checked-in `tiny_llm_ref` implementation,
and **your solution** means the code you write in `tiny_llm`. These names are
not interchangeable.

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
registers. Week 2 extensions are GPU-only: readable Python equations written
with `mlx.core` provide correctness oracles without requiring CPU BF16 support.
Vanilla Metal kernels are inspectable bring-up controls for optimized Metal
schedules, but they must still agree with the readable oracle. This contract
remains in force for Week 3, so later chapters only describe new storage and
scheduling behavior.

## Verification Status

Week 2 is an AI-assisted curriculum and systems prototype. The status of one
kind of evidence must not be used as a claim about another:

| Area | Current status | Evidence and limit |
|---|---|---|
| Curriculum sequence | Author reviewed | The checkpoint order follows the balanced measurements in the performance ledger, but the course may still be shortened. |
| Reference correctness | Continuously tested | ARM64 macOS CI builds the extensions and runs the full reference suite with Qwen3-0.6B available locally. |
| Decode-attention boundaries | Continuously tested | The suite covers Qwen head dimension 128, query lengths 1 and 8, contexts around 32 and 128, GQA ratios 1 and 4, causal masks, and explicit masks. |
| Qwen3-1.7B and Qwen3-4B integration | Optional local tests | These tests skip unless the corresponding model is already downloaded; ordinary CI does not certify them. |
| Qwen3-4B performance | One-machine evidence | Checked-in results were measured on one M4 Pro. They are research evidence, not a cross-device performance guarantee. |
| Xcode GPU profiles | Optional hardware-specific evidence | The advanced appendix defines one six-view capture contract for every Day 2–7 checkpoint. Device-specific traces explain a selected kernel but are not correctness or throughput gates. |
| Benchmark methodology | Experimental | Implementations are interleaved in balanced order and checkpoints run in fresh processes, but the results still need reproduction on more machines. |
| Maintainer kernel ownership | Incomplete | Every retained kernel still needs a maintainer pass over its invariants, winning and losing shapes, fallback, and benchmark failure modes. |

The macOS runner is a correctness gate, not a performance gate. In particular,
passing Qwen3-0.6B integration tests does not validate the Qwen3-4B throughput
story. Raw machine-specific measurements, rejected experiments, and retained
dispatch choices live in the
[performance evidence ledger](./appendix-performance.md).

## Choose a Completion Track

The full reference solution is a small performance-engineering project, not a
reasonable one-week requirement for every student. Use one of two explicit
tracks:

| Track | Required work | Provided infrastructure | Optional work |
|---|---|---|---|
| Core course | Days 1–5: cached model integration, matched benchmarking, packed W4 projections, fused model kernels, and a bounded decode-attention implementation | Model loading, extension build system, benchmark/profile runners, correctness tests, and the readable/reference implementations | Xcode counter capture, schedule searches, and hardware-specific retuning |
| Performance lab | Core course plus Days 6–7: SIMD-matrix prefill and shape-aware Split-K | Balanced operator comparisons, fresh-process progression runner, and checked-in M4 Pro evidence | Rejected experiments, alternative schedules, cross-device sweeps, and the 80%-of-MLX target |

The tests define API and correctness contracts; they do not require a student
to rediscover the reference schedule. If you are teaching the core course,
stop after Day 5 with a correct bounded kernel and treat the remaining
checkpoints as stretch work. The 80%-of-MLX acceptance target applies only to
the performance-lab track on a measured machine.

## What We Will Cover

- A dense per-request key-value cache for incremental decoding
- Synchronized benchmarking and Metal profiling of the cached baseline
- A readable quantized matrix product and a SIMD matrix-vector decode kernel
- The decode-attention primitive you implement
- Fast RMSNorm, RoPE, and SwiGLU operations
- A BF16 SIMD-matrix quantized prefill kernel
- A shape-aware split-K schedule for small Qwen prefill matrices
- A last-token output interface for generation
- An optional performance-lab target of 80% of MLX prefill and decode
  throughput on the fixed Week 2 checkpoint

Week 2 does **not** call MLX-provided implementations of the operators we are
learning. Your solution implements quantized matmul, decode attention,
RMSNorm, RoPE, and SwiGLU in its own Python, C++, or Metal code. In
particular, the completed checkpoint does not use `mx.quantized_matmul`,
`mx.dequantize`, `mx.fast` operators, or
`mx.fast.scaled_dot_product_attention` as shortcuts. The Day 1 baseline still
uses Week 1's provided `mx.dequantize` loading helper to materialize readable
dense weights; Day 3 replaces that loading path as part of keeping weights
packed.

We are still building on MLX as infrastructure. `mlx_lm` loads the official
Qwen3-4B 4-bit checkpoint and tokenizer. `mlx.core` supplies arrays, lazy graph
evaluation, memory management, device streams, and synchronization. The MLX
extension API registers your C++ primitive and dispatches your Metal kernels.
Those facilities are the platform on which your solution runs;
they are not substitutes for the operator implementations themselves.

The order is intentional:

1. **KV cache:** copy the readable Week 1 operators into a Week 2 model, add
   request-scoped state, and stop recomputing the prefix.
2. **Benchmark and profile:** measure the cached model against MLX, then rank
   real GPU costs rather than guessing what should be slow. The
   [reference-solution attribution](./appendix-performance.md#the-kernel-profile-that-selects-each-chapter)
   records the bottleneck transition that drives the remaining days.
3. **Quantized matvec:** the decode profile points at projection weight reads,
   so keep weights packed and integrate the SIMD matrix-vector kernel. Compare
   the optimized projections with MLX, then reprofile the remaining model work.
4. **Fused model kernels:** once projection latency is close to the external
   denominator, the removable decode gap moves to repeated RMSNorm, RoPE, and
   SwiGLU work. Fuse the measured cluster one operator at a time.
5. **Decode attention:** after the pointwise cluster shrinks, sweep cached
   context and measure score, softmax, and value work. Introduce online softmax
   only over its measured short-context range, then verify it with a matched
   workload that actually enters the dispatch guard.
6. **SIMD-matrix prefill:** return to the fixed 128-token workload and switch
   to its prefill profile, where the correctness-first vanilla quantized matrix
   path dominates. Introduce 8×8 matrix fragments with FP32 accumulation and
   benchmark real Qwen projection shapes.
7. **Split-K prefill:** the Day 6 shape sweep reveals under-filled grids at
   short row counts, most clearly in the narrow Qwen K/V projections. Partition
   the reduction dimension, merge BF16 partial storage with an FP32 final sum,
   and fall back to Day 6 at the measured crossover.

A later chapter never becomes an undeclared prerequisite for an earlier one.
At each boundary, the end-to-end and operator measurements select the next
kernel family. An optional Xcode trace explains what happens inside that
representative GPU kernel; the following operator profile, not the trace alone,
decides which chapter follows.

## Why FlashAttention Waits Until Week 3

Week 2 intentionally does not add dense FlashAttention. Its prefill lab first
profiles the fixed acceptance shape and follows the largest measured cost.
Week 3 then makes paged K/V the canonical serving layout, so a second
dense-only attention implementation would not become the model-facing serving
path.

Instead, Week 2 teaches the ingredients that remain useful: Day 5 introduces
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
Its paged-attention chapters combine Day 5 online softmax and Day 6 matrix
fragments only after page-table translation has been introduced, while the
quantized projections inherit Day 7's shape-aware dispatch. That boundary
lets each week's model remain understandable and runnable on its own.

The cumulative ladder is executable at any time. The
[performance appendix](./appendix-performance.md) records the matched results:

```bash
pdm run bench-week2-progression --offline --repeats 4 \
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

The default runs the reference-solution checkpoints. After implementing the
cumulative selector in your solution, add `--solution tiny_llm` to measure your
complete ladder. Preserve the named checkpoints as you work; a later
implementation should add a new branch without changing what an earlier
checkpoint executes.

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
