# Week 1 Day 7: Sampling and Preparing for Week 2

On Day 7, we will implement several sampling strategies and prepare the development environment for Week 2.

## Task 1: Sampling

On Day 6, we implemented greedy decoding. In this task, we will add temperature, top-k, and top-p (nucleus) sampling.

```
src/tiny_llm/sampler.py
```

- 📚 [mlx-lm sampler implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py)

### Temperature Sampling

When `temp=0`, use greedy decoding. When `temp` is greater than 0, sample the next token from the log-probability distribution.
A higher temperature flattens the distribution, making lower-probability tokens more likely and increasing output variety.

To implement temperature sampling, divide the log probabilities by the temperature and pass them to
`mx.random.categorical`.

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b --sampler-temp 0.5
```

### Top-k Sampling

Top-k sampling keeps only the `k` tokens with the highest log probabilities. Apply this filter before temperature scaling.

Use `mx.argpartition` to find the indices outside the top `k`, mask their log probabilities with `-mx.inf`, then apply
temperature sampling.

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b --sampler-temp 0.5 --sampler-top-k 10
```

### Top-p (Nucleus) Sampling

Top-p sampling keeps the smallest high-probability set of tokens whose cumulative probability reaches or exceeds `p`.
Apply this filter before temperature scaling.

One implementation uses `mx.argsort` to order the log probabilities from highest to lowest, applies `exp` to recover
probabilities, and applies `cumsum` to compute cumulative probability. Keep a token when the cumulative probability before
it is less than `p`; this includes the token that crosses the threshold. Mask the remaining log probabilities with
`-mx.inf`, then apply temperature sampling.

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b --sampler-temp 0.5 --sampler-top-p 0.9
```

## Task 2: Prepare for Week 2

In Week 2, we will optimize the Qwen3 serving infrastructure with C++ and Metal kernels. You will need Xcode and its
command-line tools, including the Metal compiler, to build them.

1. **Install Xcode:**

    Install Xcode from the Mac App Store or from the [Apple Developer website](https://developer.apple.com/xcode/) (this may require an Apple Developer account).

2. **Launch Xcode and Install Components:**

    After installation, launch Xcode at least once. It may prompt you to install additional macOS components; please do so (this is usually the default option).

3. **Install Xcode Command Line Tools:**

    Open your Terminal and run:

    ```bash
    xcode-select --install
    ```

4. **Set the Default Xcode Path (if needed):**

    Ensure that your command-line tools are pointing to your newly installed Xcode. You can do this by running:

    ```bash
    sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
    ```

    Adjust the path if Xcode is installed elsewhere.

5. **Accept the Xcode License:**

    You may also need to accept the Xcode license:

    ```bash
    sudo xcodebuild -license accept
    ```

6. **Install CMake:**

    ```bash
    brew install cmake
    ```

(This instruction is graciously provided by [Liu Jinyi](https://github.com/KKKZOZ).)

Test the installation by compiling the code in `src/extensions`, which contains an `axpby` function adapted from the
official MLX extension tutorial:

```bash
pdm run build-ext
pdm run build-ext-test
```

It should print `correct: True`.

If you are new to C++ or Metal, try a few small exercises before continuing. For example, implement element-wise operations
such as `exp`, `sin`, and `cos`, then use them in place of the corresponding MLX operations in your model
implementation.

That completes Week 1. We have implemented all the components required to serve Qwen3. In Week 2, we will optimize the
serving infrastructure for Apple silicon.

{{#include copyright.md}}
