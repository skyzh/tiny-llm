# tiny-llm - LLM Serving in a Week

[![CI (main)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml/badge.svg)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml)

A course on LLM serving using MLX for system engineers. The codebase
is solely (almost!) based on MLX array/matrix APIs without any high-level neural network APIs, so that we
can build the model serving infrastructure from scratch and dig into the optimizations.

The goal is to learn the techniques behind efficiently serving a large language model (e.g., Qwen3 models).

In week 1, you will implement the necessary components in Python (only Python!) to use the Qwen3 model to generate responses (e.g., attention, RoPE, etc). In week 2, **A Step Closer to vLLM**, you will add a KV cache first, then integrate and measure one course-owned Metal optimization at a time until the model reaches about 70% of MLX's performance. In week 3, **Build a Mini vLLM**, you will turn that fast model into a serving engine with continuous batching, optimized prefill, FlashAttention experiments, and paged attention. In week 4, you will use the engine to build applications such as coding agents and RAG.

Why MLX: nowadays it's easier to get a macOS-based local development environment than setting up an NVIDIA GPU.

Why Qwen3: it keeps the dense decoder architecture small enough for a local MLX course while adding modern details such as QK norm and bfloat16 weights. The official MLX 4-bit model files also make the setup predictable on Apple Silicon.

## Book

The tiny-llm book is available at [https://skyzh.github.io/tiny-llm/](https://skyzh.github.io/tiny-llm/). You can follow the guide and start building.

## Community

You may join skyzh's Discord server and study with the tiny-llm community.

[![Join skyzh's Discord Server](book/src/discord-badge.svg)](https://skyzh.dev/join/discord)

## Roadmap

Chapters substantially revised in this PR are marked as work in progress even
when they build on original course material. Unchanged chapters retain their
existing status, and all Week 4 application material remains work in progress.

| Week + Chapter | Topic                                                       | Code | Test | Doc |
| -------------- | ----------------------------------------------------------- | ---- | ---- | --- |
| 1.1            | Attention                                                   | ✅    | ✅   | ✅  |
| 1.2            | RoPE                                                        | ✅    | ✅   | ✅  |
| 1.3            | Grouped Query Attention                                     | ✅    | ✅   | ✅  |
| 1.4            | RMSNorm and MLP                                             | ✅    | ✅   | ✅  |
| 1.5            | Load the Model                                              | ✅    | ✅   | ✅  |
| 1.6            | Generate Responses (aka Decoding)                           | ✅    | ✅   | ✅  |
| 1.7            | Sampling                                                    | ✅    | ✅   | ✅  |
| 2.1            | KV Cache                                                     | ✅    | ✅   | 🚧  |
| 2.2            | Benchmarking and the MLX Baseline                            | 🚧    | 🚧   | 🚧  |
| 2.3            | Quantized Matvec                                             | ✅    | ✅   | 🚧  |
| 2.4            | Decode Attention                                             | 🚧    | 🚧   | 🚧  |
| 2.5            | Fast Model Kernels                                           | 🚧    | 🚧   | 🚧  |
| 3.1            | Continuous Batching                                           | ✅    | ✅   | 🚧  |
| 3.2            | Flash Attention for Prefill                                  | ✅    | ✅   | 🚧  |
| 3.3            | Chunked Prefill                                               | ✅    | ✅   | 🚧  |
| 3.4            | Paged KV Cache                                                | ✅    | ✅   | 🚧  |
| 3.5            | Paged Attention                                               | ✅    | ✅   | 🚧  |
| 3.6 (optional) | MoE (Mixture of Experts)                                     | ✅    | ✅   | ✅  |
| 3.7 (optional) | Performance Lab                                               | 🚧    | 🚧   | 🚧  |
| 3.8 (optional) | Speculative Decoding                                         | 🚧    | 🚧   | 🚧  |
| 4.1            | CLI Coding Agent                                              | 🚧    | 🚧   | 🚧  |
| 4.2            | RAG Pipeline                                                  | 🚧    | 🚧   | 🚧  |
| 4.3            | Tool Calling and Agent Serving                               | 🚧    | 🚧   | 🚧  |

Other topics not covered: quantized/compressed KV cache, prefix/prompt cache, fine-tuning, and long-context techniques.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=skyzh/tiny-llm&type=Date)](https://www.star-history.com/#skyzh/tiny-llm&Date)
