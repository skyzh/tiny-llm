# Week 1 Day 7: Sampling and Preparing for Week 2

The Week 1 model can now produce normalized log probabilities and generate a
greedy response. Today you will complete its single-request sampling path, run
that path in the full Qwen3 loop, and prepare the tools needed for Week 2's
custom Metal extensions.

## Task 1: Sampling

The starter already handles `temp=0` with greedy decoding. You own the
nonzero-temperature path: temperature, top-k, and top-p (nucleus) sampling.

```
src/tiny_llm/sampler.py
```

- 📚 [mlx-lm sampler implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py)

For one active request, the sampler receives normalized log probabilities with
shape `(1, vocab_size)` and returns one token-ID array with shape `(1,)` and
dtype `uint32`. The product does not pass multiple rows to this sampler.

When filtering, first copy the input. MLX indexed assignment mutates the array
object, so masking the caller's array directly would corrupt the log
probabilities that the generation loop supplied.

### Temperature Sampling

When `temp=0`, use greedy decoding. When `temp` is greater than 0, sample the next token from the log-probability distribution.
A higher temperature flattens the distribution, making lower-probability tokens more likely and increasing output variety.

To implement temperature sampling, divide the log probabilities by the
temperature and pass them to `mx.random.categorical` with `axis=-1`.

### Top-k Sampling

Top-k sampling keeps only the `k` tokens with the highest log probabilities.
Apply this filter before top-p and temperature scaling.

Use `mx.argpartition` to find the indices outside the top `k`, mask their log probabilities with `-mx.inf`, then apply
temperature sampling.

### Top-p (Nucleus) Sampling

Top-p sampling keeps the smallest high-probability set of tokens whose cumulative probability reaches or exceeds `p`.
Apply this filter before temperature scaling.

One implementation uses `mx.argsort` to order the log probabilities from highest to lowest, applies `exp` to recover
probabilities, and applies `cumsum` to compute cumulative probability. Keep a token when the cumulative probability before
it is less than `p`; this includes the token that crosses the threshold. Mask the remaining log probabilities with
`-mx.inf`.

The complete non-greedy order is:

1. copy the input;
2. apply top-k;
3. apply top-p, retaining the threshold-crossing token;
4. divide by temperature; and
5. draw with `mx.random.categorical(..., axis=-1)`.

`None` or a non-positive value disables its corresponding filter.
`top_k == vocab_size` and `top_p == 1` leave the vocabulary unfiltered. The
existing implementation raises `ValueError` when `top_k` is larger than the
vocabulary; inputs outside these boundaries are not part of this lesson's
contract.

Run the deterministic, no-download checkpoint first:

```bash
pdm run copy-test --week 1 --day 7
pdm run test --week 1 --day 7 -- -q --tb=short
```

The distribution cases observe returned-token support and frequencies; the
seeded case checks same-seed repeatability and different-seed divergence. An
unseeded draw should not be expected to return one particular token.

After the focused checkpoint passes, verify the completed sampler in the full
Week 1 product loop:

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b --sampler-temp 0.5
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b --sampler-temp 0.5 --sampler-top-k 10
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b --sampler-temp 0.5 --sampler-top-p 0.9
```

These commands require the cached 0.6B model. If it is missing, download it and
rerun them:

```bash
hf download Qwen/Qwen3-0.6B-MLX-4bit
```

Larger models remain optional and are not a Day 7 completion gate.

## Task 2: Prepare for Week 2

Week 2 Days 1 and 2 introduce KV caching in Python, so you can begin them before
the custom-extension toolchain is ready. Starting on Day 3, the C++ and Metal
work requires full Xcode, its command-line tools, the Metal compiler, and CMake
3.27 or newer.

1. **Install Xcode:**

    Install full Xcode from the Mac App Store or Apple Developer downloads.
    Full Xcode bundles its command-line tools; the standalone package described
    in [Installing the command-line tools](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools)
    is an alternative, so you do not normally need to run
    `xcode-select --install` after installing Xcode.

2. **Launch Xcode and Install Components:**

    After installation, launch Xcode at least once. It may prompt you to install additional macOS components; please do so (this is usually the default option).

3. **Verify the Active Xcode Path:**

    Check which developer directory the command-line tools use:

    ```bash
    xcode-select --print-path
    ```

    If it does not point to full Xcode, switch it as described in
    [Configuring command-line tools settings](https://developer.apple.com/documentation/xcode/configuring-command-line-tools-settings):

    ```bash
    sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
    ```

    Adjust the path if Xcode is installed elsewhere.

4. **Resolve First-Launch or License Prompts if Needed:**

    Launch Xcode once. Only if the tools report an incomplete first launch or
    license problem, follow the reported recovery step, such as:

    ```bash
    sudo xcodebuild -runFirstLaunch
    sudo xcodebuild -license accept
    ```

5. **Verify the Metal Compiler:**

    ```bash
    xcrun metal --version
    ```

    With Xcode 26, the Metal toolchain may be a separate component. If the
    command reports that it is missing, use the conditional component workflow
    described in [Downloading and installing additional Xcode components](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components),
    then verify the compiler again:

    ```bash
    xcodebuild -downloadComponent MetalToolchain
    xcrun metal --version
    ```

6. **Install and Verify CMake 3.27 or Newer:**

    ```bash
    brew install cmake
    cmake --version
    ```

(This instruction is graciously provided by [Liu Jinyi](https://github.com/KKKZOZ).)

Test the installation by compiling the code in `src/extensions`, which contains an `axpby` function adapted from the
official MLX extension tutorial:

```bash
pdm run build-ext
pdm run build-ext-test
```

It should print `c correct: True`.
The other exported extension names are fail-closed starter stubs labeled with
the Week 2 or Week 3 checkpoint that implements them; this setup check calls
only `axpby`.

If you are new to C++ or Metal, try a few small exercises before the custom
kernel work on Day 3. For example, implement element-wise operations such as
`exp`, `sin`, and `cos`, then use them in place of the corresponding MLX
operations in your model implementation.

That completes Week 1: you now have a single-request Python inference loop that
loads Qwen3, computes logits, samples tokens, and streams a response. Week 2
first adds KV caching in Python, then begins the custom Metal kernel path.

{{#include copyright.md}}
