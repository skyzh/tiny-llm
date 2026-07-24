# tiny-llm

[![CI (main)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml/badge.svg)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml)

Production inference engines are difficult to learn from. They solve model
execution, memory management, scheduling, and hardware utilization all at once.
That is the right tradeoff for serving traffic, but the wrong place to first ask
why a KV cache helps or when paging is worth its indirection.

tiny-llm takes the other route. It starts with the equations that turn Qwen3
tokens into logits, then introduces optimization and serving machinery only
when the running model gives us a reason to need it. The code stays small enough
to read end to end while still reaching the problems that shape real inference
systems: memory traffic, kernel occupancy, KV-cache growth, batching, and
request scheduling.

The course is built on MLX arrays and the MLX extension runtime, without using
high-level neural-network layers. When a chapter teaches an operator, your
solution implements that operator in Python, C++, or Metal rather than calling
the corresponding optimized MLX operation. MLX remains the correctness oracle
and performance baseline.

## The Learning Path

Each week answers a different systems question:

- **Week 1 — How does a model generate text?** Build a readable Qwen3 model
  from array operations: attention, RoPE, GQA, RMSNorm, the MLP, sampling, and
  the autoregressive loop.
- **Week 2 — Why is the readable model slow?** Add a KV cache, establish a
  synchronized MLX baseline, and let profiles choose the next optimization.
  The path moves from quantized decode matvec to fused model kernels, tiled
  prefill, and split-K where the measured Qwen shapes need it.
- **Week 3 — What changes when requests share an engine?** Introduce continuous
  batching and chunked admission, then make paged KV the canonical serving
  layout. Decode attention and FlashAttention learn to read pages directly so
  the scheduler does not rebuild dense history on every step.
- **Week 4 — What does an application demand from the inference stack?** The
  draft coding-agent week uses multi-turn sessions to motivate cache reuse,
  context compaction, rewind, interruption, and evaluation.

Week 2 follows one loop throughout:

```text
benchmark and profile -> find the next bottleneck -> optimize -> measure again
```

This matters more than any individual kernel. An optimization stays because a
whole-model benchmark supports it, not because it sounds useful in isolation.

## Why MLX and Qwen3?

Apple silicon provides a practical local environment with one shared memory
space and direct access to Metal kernels. Students can inspect the complete
path on one machine instead of depending on a remote CUDA setup.

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
appendix. The detailed chapter order and current status live in the
[book summary](book/src/SUMMARY.md).

## Project Status

Week 1 is the stable foundation. Weeks 2 and 3 are being revised around the
profile-driven optimization and continuous-serving progression. Week 4 is a
design draft and is not yet part of the rendered daily course.

## Community

Join skyzh's Discord server to study with the tiny-llm community.

[![Join skyzh's Discord Server](book/src/discord-badge.svg)](https://skyzh.dev/join/discord)
