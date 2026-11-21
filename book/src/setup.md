# Setting Up the Environment

To follow this course, you need a Mac with Apple silicon. The project uses PDM for dependency and environment management.

## Install Python

Download and run the macOS installer for
[Python 3.14.10](https://www.python.org/downloads/release/python-31410/).
After installation, `python3.14 --version` should print `Python 3.14.10`.
The lock file supports Python 3.12 through 3.14.

## Install PDM

Use PDM 2.26.7 to install the course dependencies. See the
[PDM installation guide](https://pdm-project.org/en/latest/#installation).

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
pdm use 3.14.10
pdm install -v --check --frozen-lockfile
```

This selects the interpreter for this checkout and installs the versions in
`pdm.lock`. If Python 3.14.10 is already installed and PDM cannot find it by
version, select it by absolute path. From the repository root, run the following
for the python.org installation, or substitute the path to your interpreter.

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 --version
pdm use /Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14
pdm install -v --check --frozen-lockfile
```

If you still need Python 3.14.10, run the macOS installer above and retry the
installation commands.

## Check the Installation

```bash
pdm run check-installation
```

## Run Your Tests

Start with [Week 1 Day 1](./week1-01-attention.md) immediately after the
installation check. The starter checkpoint uses small arrays from the tests.
It needs only the course Python environment.

From the repository root, run:

```bash
pdm run test --week 1 --day 1 -- -k task_1
```

This copies the Day 1 tests from `tests_refsol/` into `tests/` and runs the
focused attention checkpoint against your code in `src/tiny_llm`. On the
untouched starter, the supplied `softmax` tests pass and the attention tests
fail. Run the same command after implementing the attention task.

Run all of Day 1 with `pdm run test --week 1 --day 1`. Once you have copied
a few days, `pdm run test` reruns every test file in `tests/`. Week 1 Day 7
revisits the toolchain before the Week 2 C++ and Metal kernel lessons.

<a id="build-the-native-extensions-before-tests"></a>

## Build the Reference Extension

Your own Week 1 code is Python on top of MLX, and your `tiny_llm` package
loads its native extension only when its Week 2 tiled attention kernel is
called, so you can do Week 1's Python exercises without building the project
extension. The reference package is different: it loads its native extension
as soon as it is imported, so the reference model and reference tests below
need it built first.

Building it needs full Xcode, its Metal compiler, and CMake 3.27 or newer; the
[toolchain steps in Week 1 Day 7](./week1-07-sampling-prepare.md#task-2-prepare-for-week-2)
give the installation checks. The Python 3.14 environment was verified with
Xcode 27.2 and Metal compiler 32023.922. Then, from the repository root, run:

```bash
pdm run build-ext-ref
```

The build does not fill in the learner TODOs in `src/tiny_llm`. You build your
own extension with `pdm run build-ext` on Week 1 Day 7, before Week 2.

## Download the Model Parameters

Download the default `Qwen/Qwen3-0.6B-MLX-4bit` weights when you reach Week 1
Day 5 or want to run the reference model below. Use the Hugging Face CLI that
comes with the course environment. If your device has more memory, you can
also try the larger Qwen3 models.

```bash
pdm run hf download Qwen/Qwen3-0.6B-MLX-4bit
# Optional larger models:
pdm run hf download Qwen/Qwen3-1.7B-MLX-4bit
pdm run hf download Qwen/Qwen3-4B-MLX-4bit
```

To see the finished model generate text before building the reference
extension, use `pdm run main --solution mlx` after downloading the weights.

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

{{#include copyright.md}}
