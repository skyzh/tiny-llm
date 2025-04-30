# Setting Up the Environment

To follow along this course, you will need a mactonish device with Apple Silicon. We manage the codebase with pdm.

## Install pdm

Please follow the [offcial guide](https://pdm-project.org/en/latest/) to install pdm.

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
pdm run python check.py
# The reference solution should pass all the tests
pdm run test-week1-ref
```

## Run Unit Tests

Your code is in `src/tiny_llm`. You can run the unit tests with:

```bash
pdm run test
```

## Download the Model Parameters

We will use the Qwen2-7B-Instruct model for this course. It takes ~20GB of memory in week 1 to load the model parameters.
If you do not have enough memory, you can consider using the smaller 0.5B model. (We will make the course compatible with
it in the future; meanwhile, you have to figure out things on your own if you use the 0.5B model. Likely, this only matters
after week 1 day 6 when you start to load the model parameters.)

Follow the guide of [this page](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) to install the huggingface
cli. You should install it in your user directory/globally instead of in the tiny-llm virtual environment created by
pdm.

The model parameters are hosted on Hugging Face. Once you authenticated your cli with the credentials, you can download
them with:

```bash
# do not do this in the virtual environment created by pdm
huggingface-cli login
huggingface-cli download Qwen/Qwen2-7B-Instruct-MLX
```

Then, you can run:

```bash
pdm run main --solution week1_ref
```

It should load the model and print some text.

In week 2, we will write some kernels in C++/Metal, and we will need to set up additional tools for that. We will cover it later.

{{#include copyright.md}}
