# Setting Up the Environment

To follow this course, you need a Mac with Apple silicon. The project uses PDM for dependency and environment management.

## Install PDM

Follow the [official installation guide](https://pdm-project.org/en/latest/) to install PDM.

## Clone the Repository

```bash
git clone https://github.com/skyzh/tiny-llm
```

The repository is organized as follows:

```
src/tiny_llm/ -- your implementation
src/tiny_llm_ref/ -- the reference implementation
tests/ -- unit tests for your implementation
tests_refsol/ -- unit tests for the reference implementation
book/ -- the book source
```

Reference implementations are available if you get stuck during the course.

## Install Dependencies

```bash
cd tiny-llm
# This creates a virtual environment and installs all dependencies.
pdm install -v
```

## Check the Installation

```bash
pdm run check-installation
# The reference solution should pass all Week 1 tests.
pdm run test-refsol -- -- -k week_1
```

## Run Unit Tests

Your code is in `src/tiny_llm`. You can run the unit tests with:

```bash
pdm run test
```

## Download the Model Parameters

We use the official 4-bit Qwen3 MLX model files. The default model is `Qwen/Qwen3-0.6B-MLX-4bit`, which is small enough
for the dequantized Python implementation in Week 1. If your device has more memory, you can also try larger Qwen3 models.

Follow the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) to install the `hf`
command-line tool.

The model parameters are hosted on Hugging Face. After authenticating the CLI with your credentials, download them with:

```bash
hf auth login
hf download Qwen/Qwen3-0.6B-MLX-4bit
# Optional larger models:
hf download Qwen/Qwen3-1.7B-MLX-4bit
hf download Qwen/Qwen3-4B-MLX-4bit
```

Then, you can run:

```bash
pdm run main --solution ref --loader week1
```

The command should load the reference model and print generated text.

In Week 2, we will write C++ and Metal kernels. The required additional tools are covered at the end of Week 1.

{{#include copyright.md}}
