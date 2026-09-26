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
```

## Build the Native Extensions Before Tests

Week 1 test collection reaches the learner and reference native extensions
through package imports, even though the Day 1 attention exercise is in
Python. Before running either test command on a fresh checkout, prepare full
Xcode, its Metal compiler, and CMake 3.27 or newer; the
[toolchain steps in Week 1 Day 7](./week1-07-sampling-prepare.md#task-2-prepare-for-week-2)
give the installation checks. Then, from the repository root, build both
extensions:

```bash
pdm run build-ext
pdm run build-ext-ref
```

The reference build supplies its native import; it does not implement the
learner TODOs in `src/tiny_llm`.

Check the completed Day 1 reference exercise before working on your starter:

```bash
pdm run test-refsol --week 1 --day 1 -- -k task_1
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

Week 1 Day 7 revisits this toolchain before the Week 2 C++ and Metal kernel
lessons.

{{#include copyright.md}}
