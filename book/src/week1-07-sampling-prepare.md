# Week 1 Day 7: Sampling and Preparing for Week 2

In day 7, we will implement various sampling strategies. And we will get you prepared for week 2.

## Task 1: Sampling

We implemented the default greedy sampling strategy in the previous day. In this task, we will implement the temperature,
top-k, and top-p (nucleus) sampling strategies.

```
src/tiny_llm/sampler.py
```

- 📚 [mlx-lm sampler implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py)

**Temperature Sampling**

The first sampling strategy is the temperature sampling. When `temp=0`, we use the default greedy strategy. When it is
larger than 0, we will randomly select the next token based on the logprobs. The temperature parameter scales the distribution.
When the value is larger, the distribution will be more uniform, making the lower probability token more likely to be
selected, and therefore making the model more creative.

To implement temperature sampling, simply divide the logprobs by the temperature and use `mx.random.categorical` to
randomly select the next token.

```bash
pdm run main --solution tiny_llm --loader week1 --model Qwen/Qwen2-0.5B-Instruct-MLX --sampler-temp 0.5
```

**Top-k Sampling**

In top-k sampling, we will only keep the top-k tokens with the highest probabilities before sampling the probabilities.
This is done before the final temperature scaling.

You can use `mx.argpartition` to partition the output so that you can know the indices of the top-k elements, and then,
mask those logprobs outside the top-k with `-mx.inf`. After that, do temperature sampling.

```bash
pdm run main --solution tiny_llm --loader week1 --model Qwen/Qwen2-0.5B-Instruct-MLX --sampler-temp 0.5 --sampler-top-k 10
```

**Top-p (Nucleus) Sampling**

In top-p (nucleus) sampling, we will only keep the top-p tokens with the highest cumulative probabilities before sampling
the probabilities. This is done before the final temperature scaling.

There are multiple ways of implementing it. One way is to first use `mx.argsort` to sort the logprobs (from highest
probability to lowest), and then, do a `cumsum` over the sorted logprobs to get the cumulative probabilities. Then, mask
those logprobs outside the top-p with `-mx.inf`. After that, do temperature sampling.

```bash
pdm run main --solution tiny_llm --loader week1 --model Qwen/Qwen2-0.5B-Instruct-MLX --sampler-temp 0.5 --sampler-top-p 0.9
```

## Task 2: Prepare for Week 2

In week 2, we will optimize the serving infrastructure of the Qwen2 model. We will write some C++ code and Metal kernel
to make some operations run faster. You will need Xcode and its command-line tools, which include the Metal compiler,
to compile the C++ code and Metal kernels.

1.  **Install Xcode:**
    Install Xcode from the Mac App Store or from the [Apple Developer website](https://developer.apple.com/xcode/) (this may require an Apple Developer account).
2.  **Launch Xcode and Install Components:**
    After installation, launch Xcode at least once. It may prompt you to install additional macOS components; please do so (this is usually the default option).
3.  **Install Xcode Command Line Tools:**
    Open your Terminal and run:
    ```bash
    xcode-select --install
    ```
4.  **Set Default Xcode Path (if needed):**
    Ensure that your command-line tools are pointing to your newly installed Xcode. You can do this by running:
    ```bash
    sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
    ```
    *(Adjust the path if your Xcode is installed in a different location).*
5.  **Accept Xcode License:**
    You may also need to accept the Xcode license:
    ```bash
    sudo xcodebuild -license accept
    ```
6.  **Install CMake:**
    ```bash
    brew install cmake
    ```

(This instruction is graciously provided by [Liu Jinyi](https://github.com/KKKZOZ).)

You can test your installation by compiling the code in `src/extensions` with a `axpby` function as part of the official
mlx extension tutorial:

```bash
pdm run build-ext
pdm run build-ext-test
```

It should print `correct: True`.

If you are not familiar with C++ or Metal programming, we also suggest doing some small exercises to get familiar with
them. You can implement some element-wise operations like `exp`, `sin`, `cos` and replace the MLX ones in your model
implementation.

That's all for week 1! We have implemented all the components to serve the Qwen2 model. Now we are ready to start week 2,
where we will optimize the serving infrastructure and make it run blazing fast on your Apple Silicon device.

{{#include copyright.md}}
