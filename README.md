# tiny-llm

[![CI (main)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml/badge.svg)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml)

tiny-llm is a hands-on course for systems engineers who want to understand LLM
inference end to end. You can think of it as an LLM-serving counterpart to
CMU's [Needle](https://github.com/dlsyscourse/hw1/tree/main/python/needle)
project: build the path that loads a Qwen3 model, turns tokens into logits, and
generates text.

The course begins with array and matrix operations, then introduces kernels and
serving machinery as the running model needs them. Keeping the implementation
small enough to read end to end makes it possible to connect the equations to
memory traffic, kernel occupancy, KV-cache growth, batching, and request
scheduling.

The course is built on MLX arrays and the MLX extension runtime, without using
high-level neural-network layers. When a chapter teaches an operator, your
solution implements that operator in Python, C++, or Metal rather than calling
the corresponding optimized MLX operation. MLX remains the correctness oracle
and performance baseline.

## The Learning Path

The course follows a four-week learning path:

- **Week 1: From Matmul to Text.** Build a Qwen3 model directly from `mlx.core`
  array operations: attention, RoPE, GQA, RMSNorm, the MLP, sampling, and
  the autoregressive loop.
- **Week 2: A Step Closer to vLLM.** Add a KV cache, establish a
  synchronized MLX baseline, and let matched benchmarks choose each optimization.
  The path moves from quantized decode matvec to fused model kernels, tiled
  prefill, and split-K where the measured Qwen shapes need it.
- **Week 3: Build a Mini vLLM.** Introduce continuous
  batching and chunked admission, then make paged KV the canonical serving
  layout. Decode attention and FlashAttention learn to read pages directly so
  the scheduler does not rebuild dense history on every step.
- **Week 4: Build a Coding Agent.** Start with a bounded, validated agent loop,
  then connect it to a small workspace. The course is publishing one reviewed
  checkpoint at a time; Days 1 through 5 now cover inspection, approved edits,
  one validation command, simple effect receipts, and one visible
  checkpoint-and-resume boundary, and receipt-backed context compaction.

## Why MLX and Qwen3?

Apple silicon provides a practical local environment with one shared memory
space and direct access to Metal kernels. Students can inspect the complete
path on one machine instead of depending on an expensive CUDA GPU setup.

Qwen3-4B is large enough to expose real weight-bandwidth, attention, and cache
costs, but small enough to iterate on locally. Its grouped-query attention,
QK normalization, BF16 activations, and 4-bit weights also keep the exercises
close to current model-serving work.

## Start Here

The book is published at
[skyzh.github.io/tiny-llm](https://skyzh.github.io/tiny-llm/). Begin with the
[environment setup](https://skyzh.github.io/tiny-llm/setup.html), or verify an
existing checkout with:

```bash
pdm install -v
pdm run check-installation
pdm run test-refsol -- -- -k week_1
```

The `tiny_llm` package is where students implement the exercises.
`tiny_llm_ref` contains the reference solution used by the tests and benchmark
appendix. The [book summary](book/src/SUMMARY.md) lists the chapter order;
implementation, test, and publication readiness is tracked below.

## Roadmap

The table tracks implementation (`Code`), tests (`Test`), rendered chapters (`Doc`), and Chi's review of learner-facing material (`Audit`). Week 4 is publishing one reviewed day at a time; Days 1 through 5 are currently available to learners. The Audit column reflects Chi's personal editorial pass on the published course content and is independent of code/test/doc readiness.

Day 3 can send file contents to the model, modify files after approval, and run
one exact configured command. Use a disposable workspace without secrets and
read the [Week 4 overview](book/src/week4-overview.md) before running the loop.
Day 4 checkpoints a complete tool-observation boundary with the scripted
model's fake cache metadata, then restores a fresh model without replaying the
completed edit or command.
Day 5 compacts older completed effects in the model-visible transcript while
their exact receipts retain the full action, result, and changed artifacts.

| Week + Chapter | Topic | Code | Test | Doc | Audit |
|---|---|---|---|---|---|
| 1.1 | Attention | ✅ | ✅ | ✅ | ✅ |
| 1.2 | RoPE | ✅ | ✅ | ✅ | ✅ |
| 1.3 | Grouped Query Attention | ✅ | ✅ | ✅ | ✅ |
| 1.4 | RMSNorm and MLP | ✅ | ✅ | ✅ | ✅ |
| 1.5 | Load the Model | ✅ | ✅ | ✅ | ✅ |
| 1.6 | Generate Responses (aka Decoding) | ✅ | ✅ | ✅ | ✅ |
| 1.7 | Sampling | ✅ | ✅ | ✅ | ✅ |
| 2.1 | KV Cache | ✅ | ✅ | ✅ | 🚧 |
| 2.2 | Benchmarking and Profiling | ✅ | ✅ | ✅ | 🚧 |
| 2.3 | Quantize the Model | ✅ | ✅ | ✅ | 🚧 |
| 2.4 | Fused Model Kernels | ✅ | ✅ | ✅ | 🚧 |
| 2.5 | Fused Decode Attention | ✅ | ✅ | ✅ | 🚧 |
| 2.6 | SIMD-Matrix Prefill | ✅ | ✅ | ✅ | 🚧 |
| 2.7 | Split-K Prefill | ✅ | ✅ | ✅ | 🚧 |
| 3.1 | Continuous Batching | ✅ | ✅ | ✅ | 🚧 |
| 3.2 | Chunked Prefill | ✅ | ✅ | ✅ | 🚧 |
| 3.3 | Paged KV Cache | ✅ | ✅ | ✅ | 🚧 |
| 3.4 | Direct Paged Attention | ✅ | ✅ | ✅ | 🚧 |
| 3.5 | Paged FlashAttention | ✅ | ✅ | ✅ | 🚧 |
| 3.6 (optional) | Speculative Decoding | ✅ | ✅ | ✅ | 🚧 |
| 3.x (optional) | MoE (Mixture of Experts) | ✅ | ✅ | ✅ | 🚧 |
| 4.1 | Validated Agent Loop | ✅ | ✅ | ✅ | 🚧 |
| 4.2 | Inspect a Workspace | ✅ | ✅ | ✅ | 🚧 |
| 4.3 | Edit, Validate, and Record | ✅ | ✅ | ✅ | 🚧 |
| 4.4 | Checkpoint and Resume | ✅ | ✅ | ✅ | 🚧 |
| 4.5 | Compact Completed Work | ✅ | ✅ | ✅ | 🚧 |
| 4.6–4.7 | Later agent checkpoints | — | — | — | 🚧 |

Other topics not covered include quantized or compressed KV caches,
cross-request prefix caching, fine-tuning, and long-context techniques.

## Community

Join skyzh's Discord server to study with the tiny-llm community.

[![Join skyzh's Discord Server](book/src/discord-badge.svg)](https://skyzh.dev/join/discord)
