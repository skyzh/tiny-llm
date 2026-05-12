# Setting Up the Environment

To follow along this course, you will need a Macintosh device with Apple Silicon. We manage the codebase with pdm.

## Install pdm

Please follow the [official guide](https://pdm-project.org/en/latest/) to install pdm.

## Clone the Repository

```bash
git clone https://github.com/skyzh/tiny-llm
```

The repository is organized as follows:

```
src/tiny_llm -- your implementation
src/tiny_llm_week1_ref -- reference implementation of week 1
tests/ -- unit tests for your implementation
tests_ref_impl_week1/ -- unit tests for the reference implementation of week 1
book/ -- the book
```

We provide all reference implementations and you can refer to them if you get stuck in the course.

## Install Dependencies

```bash
cd tiny-llm
pdm install -v # this will automatically create a virtual environment and install all dependencies
```

## Check the Installation

```bash
pdm run check-installation
# The reference solution should pass all the *week 1* tests
pdm run test-refsol -- -- -k week_1
```

## Run Unit Tests

Your code is in `src/tiny_llm`. You can run the unit tests with:

```bash
pdm run test
```

## Download the Model Parameters

We will use the official Qwen3 MLX 4-bit checkpoints for this course. The default model is `Qwen/Qwen3-0.6B-MLX-4bit`, which is small enough for the Week 1 dequantized Python implementation. If you have more memory, you can also try the larger Qwen3 MLX checkpoints.

Follow the guide of [this page](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) to install the Hugging Face
CLI (`hf`).

The model parameters are hosted on Hugging Face. Once you authenticated your cli with the credentials, you can download
them with:

```bash
hf login
hf download Qwen/Qwen3-0.6B-MLX-4bit
hf download Qwen/Qwen3-1.7B-MLX-4bit
hf download Qwen/Qwen3-4B-MLX-4bit
```

Then, you can run:

```bash
pdm run main --solution ref --loader week1
```

It should load the model and print some text.

In week 2, we will write some kernels in C++/Metal, and we will need to set up additional tools for that. We will cover it later.

{{#include copyright.md}}
