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

<a id="build-the-native-extensions-before-tests"></a>

## Build the Reference Extension

Your own Week 1 code is plain Python, and your `tiny_llm` package loads its
native extension only when its Week 2 tiled attention kernel is called, so you
can do every Week 1 exercise without building anything. The reference package
is different: it loads its native extension as soon as it is imported, so the
reference model and reference tests below need it built first.

Building it needs full Xcode, its Metal compiler, and CMake 3.27 or newer; the
[toolchain steps in Week 1 Day 7](./week1-07-sampling-prepare.md#task-2-prepare-for-week-2)
give the installation checks. Then, from the repository root, run:

```bash
pdm run build-ext-ref
```

The build does not fill in the learner TODOs in `src/tiny_llm`. You build your
own extension with `pdm run build-ext` on Week 1 Day 7, before Week 2.

If you do not have the toolchain yet, you can still start. Download the model,
skip [Run the Reference Model](#run-the-reference-model), and go on to
[Run Your Tests](#run-your-tests). To see the finished model generate text
without any native build, run `pdm run main --solution mlx`, which uses plain
MLX instead of the reference implementation.

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

## Run the Reference Model

Now run the model you are going to build:

```bash
pdm run main --solution ref --loader week1
```

The command loads the reference implementation and prints generated text. By
the end of Week 1 Day 6, the same command with `--solution tiny_llm` will run
your own implementation instead.

The reference tests should also pass. The Week 1 Day 5 tests load the 0.6B
model, so run them after the download:

```bash
pdm run test-refsol --week 1 --day 1 -- -k task_1
# The reference solution should pass all Week 1 tests.
pdm run test-refsol -- -- -k week_1
```

## Run Your Tests

Your code is in `src/tiny_llm`, and its tests start out absent: `tests/` holds
only shared helpers. Each day's command copies that day's tests from
`tests_refsol/` into `tests/` and runs them:

```bash
pdm run test --week 1 --day 1
```

On the untouched starter, the supplied `softmax` tests pass and the attention
and `linear` tests fail. That red result is your starting point for
[Week 1 Day 1](./week1-01-attention.md). Once you have copied a few days,
`pdm run test` with no arguments reruns every test file in `tests/`.

Week 1 Day 7 revisits the toolchain before the Week 2 C++ and Metal kernel
lessons.

{{#include copyright.md}}
