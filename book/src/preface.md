# Learn LLM Serving

This course is designed for systems engineers who want to understand how large language models (LLMs) work.

As a systems engineer, I am always curious about how things work internally and how to optimize them. I found it difficult
to understand LLM inference because most open-source serving projects are highly optimized with CUDA kernels and other
low-level techniques. It is hard to see the whole picture in a codebase with hundreds of thousands of lines. I therefore
decided to implement an LLM serving project from scratch using only array and matrix operations. The goal was to understand
what it takes to load an LLM's parameters and perform the mathematical operations that generate text.

You can think of this course as an LLM counterpart to the [Needle](https://github.com/dlsyscourse/hw1/tree/main/python/needle)
project from CMU's Deep Learning Systems course.

## Prerequisites

You should understand the basics of deep learning and be familiar with PyTorch. We recommend the following resources:

- CMU [Introduction to Machine Learning](https://www.cs.cmu.edu/~mgormley/courses/10601/) — covers the fundamentals of machine learning.
- CMU [Deep Learning Systems](https://dlsyscourse.org) — teaches you how to build a framework like PyTorch from scratch.

## Environment Setup

This course uses [MLX](https://github.com/ml-explore/mlx), an array and machine learning framework for Apple silicon. For
many learners, an Apple silicon device is easier to access than an NVIDIA GPU. In principle, you could also complete the
course with PyTorch or NumPy, but the test infrastructure does not support them as implementation backends. Instead, the
tests compare your implementation with trusted MLX operations and model implementations to verify correctness.

## Course Structure

This course is divided into four weeks. We will serve Qwen3 MLX models, optimize the serving path, and use it to build a
small coding agent.

- Week 1: Serve Qwen3 using array and matrix operations written in Python.
- Week 2: Day 1 caches a request prefix and bounds its dense storage. Day 2
  keeps W4 projection weights packed in the cached model. Day 3 adds a SIMD
  matrix prefill path. Day 4 integrates fused RMSNorm, RoPE, and SwiGLU one
  checkpoint at a time. Tiled prefill attention remains a later lesson.
- Week 3: Add further optimizations and batch requests for high-throughput serving.
- Week 4: Reuse the serving stack in a local coding agent with tools, sessions, and evaluation.

## Course Roadmap: What Depends on What

The course supports two different goals: **implementing** the cumulative
serving stack, or **studying and running** a later checkpoint without completing
all earlier exercises. These are not the same path.

The current route runs from Week 1 through [Week 2 Day 1](./week2-01-kv-cache.md),
[Day 2](./week2-02-quantize-model.md), [Day 3](./week2-03-simd-matrix-prefill.md),
then [Day 4](./week2-04-fused-model-kernels.md): `kv-cache`, `capacity-cache`,
`quantized-matvec`, `simd-matmul`, `rmsnorm`, `rope`, then `swiglu`.
The tiled-prefill Day 5 checkpoint is planned, so this four-day checkout does
not offer a completed Week 2 → Week 3 learner path. The
[earlier full-course roadmap diagram](./course-roadmap.svg) is
retained as historical context; its seven-day Week 2 order is not this
checkout's navigation.

The reference and full-MLX models are useful controls, but they do not fill
unfinished functions in `src/tiny_llm`. A later custom kernel may have an MLX
operator substitution at the same interface; Day 1's cache state has no such
operator shortcut.

| Your goal | Start here | What earlier implementation is required? |
| --- | --- | --- |
| Build the currently shipped cache path | Week 1, then Week 2 Day 1 | Implement both cache checkpoints and their matched measurement. |
| Build the current packed-weight path | Complete Day 1, then Week 2 Day 2 | Keep the capacity cache and implement the packed operator and model wiring. |
| Build the current SIMD prefill path | Complete Days 1 and 2, then Week 2 Day 3 | Keep the bounded cache and packed weights; implement the SIMD matrix tile and model wiring. |
| Build the current fused-primitives path | Complete Days 1–3, then Week 2 Day 4 | Keep the cached packed model and integrate RMSNorm, RoPE, and SwiGLU in order. |
| Study an older Week 2 experiment | Read its historical page | Its former day numbers and commands are not current gates. |
| Read or experiment with a later week | Open that chapter and use `tiny_llm_ref` | None in your learner tree. Run the supplied reference tests or reference loader. |
| Compare with the production-library baseline | Use `--solution mlx` | None, but this runs the full MLX model and bypasses the course implementation. |
| Run the Week 4 Days 1–7 deterministic tests before finishing the serving stack | After setup, run the supplied scripted-model tests | The tests do not need a working serving implementation. The course still assumes setup plus Weeks 1–3 before Week 4; follow the Week 4 days in order, and Day 8's real-model bridge needs the Week 3 model/tokenizer/KV-cache boundary. |

The cumulative dependencies are deliberate:

- **Week 1 → Week 2:** Day 1 keeps the readable model, adds request-owned
  dense K/V reuse and bounded storage, and compares the same request at both
  cache checkpoints. Day 2 keeps W4 weights packed through the live model.
  Day 3 adds a SIMD matrix path for prefill while keeping that model state.
  Day 4 adds three cumulative fused primitives around those projections.
- **Week 2 → Week 3:** Week 3 selects MLX quantized projections, but it keeps
  course-owned normalization, activation, cache, attention, paging, batching,
  and scheduling. This is an explicit operator seam, not “use the MLX model for
  Week 2.”
- **Week 3 → Week 4:** Week 4 remains the next cumulative course week. Its
  Days 1–7 tests can exercise control flow with deterministic scripted models
  after setup, even before the serving stack works. Day 8 reconnects that
  harness to the real tokenizer and KV cache, so that checkpoint needs a
  working Week 3 path.

> **Is Week 2 required for Week 3? Its interfaces are; every future
> optimization is not.** The current Week 3 starter reuses the Week 2 model
> shell, dense-cache contract, packed-weight plumbing, normalization,
> activation, attention, and matrix-fragment interfaces. You may preserve
> those interfaces and substitute MLX operators for custom optimization work,
> but starting Week 3 is **not as simple as selecting `--solution mlx`**. That
> flag selects the complete MLX model and bypasses the course-owned paging,
> batching, attention, and scheduler surfaces that Week 3 teaches. Skipping
> the entire Week 2 implementation would require a supplied hybrid starting
> checkpoint; that checkpoint does not exist today.

### Later Week 2 operator off-ramps

Day 1 requires cache state and matched measurement; it has no replaceable
custom kernel. Days 2 and 3 have an optional `mx.quantized_matmul` substitution
at the projection operator boundary; both still need the course-owned cache
and model wiring. Day 4 can substitute the equivalent MLX operator or
equation at one RMSNorm, RoPE, or SwiGLU boundary while preserving the other
course-owned paths. The earlier full-course book retains additional
mechanisms and old addresses in the
[historical Week 2 pages](./week2-02-benchmark-profile.md). Selecting
`--solution mlx` runs a separate complete model, not a hybrid that completes
learner cache TODOs.

To run a completed checkpoint without solving it first:

```bash
# Run the Day 1 reference tests after setup.
pdm run test-refsol --week 2 --day 1

# Run a completed course model.
pdm run main --solution tiny_llm_ref --loader week2 --week2-checkpoint kv-cache

# Run the separate full-MLX baseline.
pdm run main --solution mlx
```

The Day 1 reference tests do not require `pdm run build-ext-ref`. Build the
reference extension for Days 2–4's native tests, alongside
the learner extension; neither reference solution fills your learner TODOs.

`--solution tiny_llm_ref` runs the supplied implementation end to end. `--solution mlx`
runs MLX end to end. Neither command composes “earlier weeks from the reference
or MLX, this week's TODOs from my learner tree.” Per-operator substitution is a
manual code edit that preserves the course interface; it is not a third
solution mode. If you want to implement a later week in `src/tiny_llm`, its
earlier interface and state prerequisites must already work; the repository
does not currently provide a one-command hybrid checkpoint.

## Choose a Model for Your Mac

The table below is a conservative starting point for recent Apple-silicon Mac mini and MacBook unified-memory sizes up to
64 GB. Across those machines, the available tiers are 8, 16, 18, 24, 32, 36, 48, and 64 GB.[^mac-memory-tiers] Each entry
is **recommended / maximum** for that week's course path. The recommendation is the checkpoint to use while completing
the exercises; the maximum is the largest course-supported checkpoint worth trying with short prompts and the chapter's
default batch settings.

| Unified memory | Week 1 | Week 2 | Week 3 | Week 4 |
| --- | --- | --- | --- | --- |
| 8 GB | 0.6B / 0.6B | 0.6B / 0.6B[^week2-dense] | 0.6B / 1.7B | 0.6B / 1.7B |
| 16 GB | 0.6B / 1.7B | 0.6B / 1.7B[^week2-dense] | 4B / 8B | 4B / 8B |
| 18 GB | 0.6B / 1.7B | 0.6B / 1.7B[^week2-dense] | 4B / 8B | 4B / 8B |
| 24 GB | 0.6B / 1.7B | 0.6B / 1.7B[^week2-dense] | 4B / 8B | 4B / 8B |
| 32 GB | 4B / 8B | 4B / 8B | 4B / 30B-A3B[^moe] | 4B / 30B-A3B[^moe] |
| 36 GB | 4B / 8B | 4B / 8B | 4B / 30B-A3B[^moe] | 4B / 30B-A3B[^moe] |
| 48 GB | 4B / 8B | 4B / 8B | 4B / 30B-A3B[^moe] | 4B / 30B-A3B[^moe] |
| 64 GB | 4B / 8B | 4B / 8B | 4B / 30B-A3B[^moe] | 4B / 30B-A3B[^moe] |

Week 1 reads an official 4-bit checkpoint but materializes its linear and embedding weights in BF16. On an 8 GB Mac,
keep the required path at 0.6B. On a 16–24 GB Mac, use 0.6B for the required work and treat 1.7B as an upper-end experiment.
Week 2 Day 1 retains that dense BF16 model. In this checkout, Day 2 keeps
projection weights packed for `quantized-matvec`, and Day 3 reuses those
weights for `simd-matmul` prefill. Day 4 keeps the packed path while adding
RMSNorm, RoPE, and SwiGLU; the Week 3 and 4 paths expect the packed interface.
More memory still helps after reaching the largest
supported model because prompt length, batch size, KV caches, compilation, macOS, and other applications all share the
same pool. These ceilings are therefore planning guidance, not a guarantee that every workload will avoid memory
pressure.

[^mac-memory-tiers]: Apple lists these tiers across the [M2 Mac mini](https://support.apple.com/en-us/111837),
    [M3 Pro and M3 Max MacBook Pro](https://support.apple.com/en-us/117736),
    [M4 Mac mini](https://support.apple.com/en-us/121555), and
    [M5 MacBook Air](https://support.apple.com/en-us/126320) specifications. Higher-memory configurations are outside
    this table.
[^week2-dense]: These conservative Week 2 model-size choices cover Day 1's
    dense BF16 checkpoint. Days 2–4 retain the cache and keep projection
    weights packed. Use 0.6B for the required Day 3 and Day 4 comparisons.
    The optional packed 4B comparison is described in
    [Day 2](./week2-02-quantize-model.md); its matched Day 1 control still
    needs enough memory for the dense `capacity-cache` model.
[^moe]: 30B-A3B requires the optional Week 3 MoE implementation. In Week 4, select the Week 3 loader. Use batch size one
    and a short context when approaching this ceiling; 4B remains the required-course target.

## How to Use This Book

The tiny-llm book is a hands-on guide rather than a textbook that explains every concept from first principles. We link
to the resources that the authors found useful while implementing the project instead of repeating their explanations.
Each chapter provides a sequence of tasks, supporting readings, and implementation hints.

The book also standardizes terminology and notation across those resources so that they map cleanly to the codebase. For
example, we use consistent symbols for tensor dimensions and explain what `H`, `L`, and `E` mean at the point of use.

## About the Authors

This course is created by [Chi](https://github.com/skyzh) and [Connor](https://github.com/Connor1996).

Chi is a systems software engineer at [Neon](https://neon.tech) (now acquired by Databricks), focusing on storage systems.
Fascinated by large language models, he created this course to explore how LLM inference works.

Connor is a software engineer at [PingCAP](https://pingcap.com), developing the TiKV distributed key-value database.
Curious about the internals of LLMs, he joined the project to practice building a high-performance LLM serving system
from scratch and helped develop the course for the community.

## Community

You can join skyzh's Discord server to study with the tiny-llm community.

[![Join skyzh's Discord Server](discord-badge.svg)](https://skyzh.dev/join/discord)

## Get Started

Follow the instructions in [Setting Up the Environment](./setup.md), then begin building tiny-llm.

{{#include copyright.md}}
