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

## Choose a Model for Your Mac

The table below is a conservative starting point for common MacBook unified-memory sizes. Each entry is
**recommended / maximum** for that week's course path. The recommendation is the checkpoint to use while completing the
exercises; the maximum is the largest course-supported checkpoint worth trying with short prompts and the chapter's
default batch settings.

| Unified memory | Week 1 | Week 2 | Week 3 | Week 4 |
| --- | --- | --- | --- | --- |
| 16 GB | 0.6B / 1.7B | 4B / 8B[^week2-dense] | 4B / 8B | 4B / 8B |
| 32 GB | 4B / 8B | 4B / 8B | 4B / 30B-A3B[^moe] | 4B / 30B-A3B[^moe] |
| 64 GB | 4B / 8B | 4B / 8B | 4B / 30B-A3B[^moe] | 4B / 30B-A3B[^moe] |

Week 1 reads an official 4-bit checkpoint but materializes its linear and embedding weights in BF16. On a 16 GB Mac,
use 0.6B for the required work and treat 1.7B as an upper-end experiment. Week 2 Days 1–2
retain that dense BF16 model; Day 3 keeps weights packed for the quantized-matvec checkpoint. Weeks 3
and 4 inherit that packed path. More memory still helps after reaching the largest
supported model because prompt length, batch size, KV caches, compilation, macOS, and other applications all share the
same pool. These ceilings are therefore planning guidance, not a guarantee that every workload will avoid memory
pressure.

[^week2-dense]: Week 2 Days 1–2 use the dense Week 1 loader. On a 16 GB Mac, keep using 0.6B
    until the packed quantized-matvec path is complete on Day 3; the 4B recommendation and 8B maximum
    apply after that checkpoint.
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
