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
- Week 2: Implement custom C++ and Metal kernels to accelerate the model.
- Week 3: Add further optimizations and batch requests for high-throughput serving.
- Week 4: Reuse the serving stack in a local coding agent with tools, sessions, and evaluation.

## Course Roadmap: What Depends on What

The course supports two different goals: **implementing** the cumulative
serving stack, or **studying and running** a later checkpoint without completing
all earlier exercises. These are not the same path.

<style>
  .course-roadmap-scroll {
    max-width: 100%;
    overflow-x: auto;
    padding-bottom: 0.25rem;
    overscroll-behavior-x: contain;
  }
  .course-roadmap-scroll:focus-visible {
    outline: 2px solid currentColor;
    outline-offset: 3px;
  }
  .course-roadmap-scroll img {
    display: block;
    width: 1200px;
    min-width: 1200px;
    max-width: none;
    height: auto;
  }
</style>

<div id="course-roadmap-scroll" class="course-roadmap-scroll" role="region" aria-label="Scrollable Tiny-LLM course roadmap" aria-describedby="course-roadmap-scroll-help" tabindex="0">
  <img src="./course-roadmap.svg" alt="Tiny-LLM roadmap. The cumulative interface and state path runs from Week 1 through the seven Week 2 days and Week 3 into Week 4. Week 2 Days 3 through 7 show optional MLX operator off-ramps that preserve the course interfaces; they are different from the full-MLX model baseline. Week 4 keeps the course prerequisite of setup plus Weeks 1 through 3, while its deterministic scripted-model tests for Days 1 through 7 can run after setup. Day 8 joins the scripted sequence to the real-model path, and Day 9 continues the Week 4 sequence.">
</div>

<p id="course-roadmap-scroll-help">On a narrow screen, scroll the roadmap
horizontally; when the roadmap is focused, the left and right arrow keys move
through it without changing chapters. Its labels stay at their readable desktop
size.</p>

<script>
  (() => {
    const roadmap = document.getElementById("course-roadmap-scroll");
    roadmap.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      event.stopPropagation();
      const distance = Math.max(120, Math.round(roadmap.clientWidth * 0.8));
      roadmap.scrollLeft += event.key === "ArrowLeft" ? -distance : distance;
    });
  })();
</script>

Solid arrows in the diagram are **interface and state prerequisites**. They do
not mean that you must hand-write every earlier optimization. A dashed border
marks a custom operator that you may replace locally with its MLX equivalent
while keeping the surrounding course interface. The reference and full-MLX
lanes let you observe a completed system, but they do not fill in unfinished
functions in `src/tiny_llm`.

| Your goal | Start here | What earlier implementation is required? |
| --- | --- | --- |
| Build the whole serving system | Week 1, then follow the solid arrows | Each week uses interfaces and mechanisms established by the previous week. |
| Skip a Week 2 kernel optimization | Keep that day's course interface and wire the corresponding MLX operator at the seam | The earlier model, state, and interface work still needs to exist. This is a local code choice, not a CLI flag. |
| Read or experiment with a later week | Open that chapter and use `tiny_llm_ref` | None in your learner tree. Run the supplied reference tests or reference loader. |
| Compare with the production-library baseline | Use `--solution mlx` | None, but this runs the full MLX model and bypasses the course implementation. |
| Run the Week 4 Days 1–7 deterministic tests before finishing the serving stack | After setup, run the supplied scripted-model tests | The tests do not need a working serving implementation. The course still assumes setup plus Weeks 1–3 before Week 4; follow the Week 4 days in order, and Day 8's real-model bridge needs the Week 3 model/tokenizer/KV-cache boundary. |

The cumulative dependencies are deliberate:

- **Week 1 → Week 2:** Week 2 starts from the readable Qwen3 model and replaces
  costs one mechanism at a time: first the generation algorithm and KV cache,
  then quantized and fused kernels. Days 1–2 establish state and measurement;
  Days 3–7 expose optimization seams.
- **Week 2 → Week 3:** Week 3 selects MLX quantized projections, but it keeps
  course-owned normalization, activation, cache, attention, paging, batching,
  and scheduling. This is an explicit operator seam, not “use the MLX model for
  Week 2.”
- **Week 3 → Week 4:** Week 4 remains the next cumulative course week. Its
  Days 1–7 tests can exercise control flow with deterministic scripted models
  after setup, even before the serving stack works. Day 8 reconnects that
  harness to the real tokenizer and KV cache, so that checkpoint needs a
  working Week 3 path.

> **Is Week 2 required for Week 3? The Week 2 interfaces are; every Week 2
> optimization is not.** The current Week 3 starter reuses the Week 2 model
> shell, dense-cache contract, packed-weight plumbing, normalization,
> activation, attention, and matrix-fragment interfaces. You may preserve
> those interfaces and substitute MLX operators for custom optimization work,
> but starting Week 3 is **not as simple as selecting `--solution mlx`**. That
> flag selects the complete MLX model and bypasses the course-owned paging,
> batching, attention, and scheduler surfaces that Week 3 teaches. Skipping
> the entire Week 2 implementation would require a supplied hybrid starting
> checkpoint; that checkpoint does not exist today.

### Week 2 operator off-ramps

Week 2 separates the mechanism you need later from the kernel you are invited
to optimize. If your goal is to continue into Week 3 rather than implement
every Metal kernel, you can make these explicit local substitutions:

| Week 2 day | Keep in the course stack | Optional MLX substitution |
| --- | --- | --- |
| Days 1–2 | Dense KV-cache state, the Week 2 model boundary, and the matched measurement method | None; these are state and methodology rather than replaceable operators. |
| Day 3 | Packed-weight containers, quantized embedding/model wiring, and the `quantized_linear` interface | Route projections through `mx.quantized_matmul` instead of the custom matrix-vector kernel. |
| Day 4 | The Week 2 norm, position, and activation call sites | Use the corresponding MLX RMSNorm/RoPE operators and an MLX SiLU-based SwiGLU composition instead of the custom fused kernels. |
| Day 5 | The dense-cache attention interface and its shape/mask adapter | Use `mx.fast.scaled_dot_product_attention` instead of the custom decode-attention kernel. |
| Days 6–7 | The same quantized-projection interface and dispatch boundary | Keep using the Day 3 MLX projection seam instead of implementing SIMD-matrix and Split-K schedules. |

Only the quantized-projection seam is already selected by canonical Week 3.
The Day 4 and Day 5 alternatives require you to wire the MLX call at the
existing course interface; there is no `--use-mlx-for-day` command. These
off-ramps let you study later mechanisms, but they do not complete the skipped
day's custom-kernel exercises, implementation-specific tests, or performance
claims.

To run a completed checkpoint without solving it first:

```bash
# Run one supplied reference test group.
pdm run test-refsol --week 3 --day 1

# Run a completed course model.
pdm run main --solution ref --loader week3

# Run the separate full-MLX baseline.
pdm run main --solution mlx
```

`--solution ref` runs the supplied implementation end to end. `--solution mlx`
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
| 8 GB | 0.6B / 0.6B | 0.6B / 1.7B[^week2-dense] | 0.6B / 1.7B | 0.6B / 1.7B |
| 16 GB | 0.6B / 1.7B | 4B / 8B[^week2-dense] | 4B / 8B | 4B / 8B |
| 18 GB | 0.6B / 1.7B | 4B / 8B[^week2-dense] | 4B / 8B | 4B / 8B |
| 24 GB | 0.6B / 1.7B | 4B / 8B[^week2-dense] | 4B / 8B | 4B / 8B |
| 32 GB | 4B / 8B | 4B / 8B | 4B / 30B-A3B[^moe] | 4B / 30B-A3B[^moe] |
| 36 GB | 4B / 8B | 4B / 8B | 4B / 30B-A3B[^moe] | 4B / 30B-A3B[^moe] |
| 48 GB | 4B / 8B | 4B / 8B | 4B / 30B-A3B[^moe] | 4B / 30B-A3B[^moe] |
| 64 GB | 4B / 8B | 4B / 8B | 4B / 30B-A3B[^moe] | 4B / 30B-A3B[^moe] |

Week 1 reads an official 4-bit checkpoint but materializes its linear and embedding weights in BF16. On an 8 GB Mac,
keep the required path at 0.6B. On a 16–24 GB Mac, use 0.6B for the required work and treat 1.7B as an upper-end experiment.
Week 2 Days 1–2
retain that dense BF16 model; Day 3 keeps weights packed for the quantized-matvec checkpoint. Weeks 3
and 4 inherit that packed path. More memory still helps after reaching the largest
supported model because prompt length, batch size, KV caches, compilation, macOS, and other applications all share the
same pool. These ceilings are therefore planning guidance, not a guarantee that every workload will avoid memory
pressure.

[^mac-memory-tiers]: Apple lists these tiers across the [M2 Mac mini](https://support.apple.com/en-us/111837),
    [M3 Pro and M3 Max MacBook Pro](https://support.apple.com/en-us/117736),
    [M4 Mac mini](https://support.apple.com/en-us/121555), and
    [M5 MacBook Air](https://support.apple.com/en-us/126320) specifications. Higher-memory configurations are outside
    this table.
[^week2-dense]: Week 2 Days 1–2 use the dense Week 1 loader, so keep using the Week 1 recommendation
    until the packed quantized-matvec path is complete on Day 3. The larger Week 2 entries apply after that checkpoint.
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
