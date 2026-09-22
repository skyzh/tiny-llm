# 🚧 Week 2 Day 5: Compact Model Primitives

At `simd-matmul`, projections use packed weights and shape-aware schedules, but
normalization, position rotation, and the MLP activation remain readable MLX
expressions. Day 5 replaces them one at a time so every change has its own
cumulative checkpoint and fallback:

`simd-matmul` → `rmsnorm` → `rope` → `swiglu`.

Your seams are the wrappers in `src/tiny_llm/week2_kernels.py`, the native
primitive boundary, the Metal kernels, and the corresponding model switches.

## First Diagnostic: Keep RMSNorm Values in Registers

```bash
pdm run build-ext
pdm run test --week 2 --day 5 -- -k register_cached
```

The intended first failure reaches `FastRMSNorm` or its register-cached kernel.
For a row `x` with learned weight `w`:

$$
\operatorname{RMSNorm}(x)_i =
\frac{x_i}{\sqrt{\frac{1}{D}\sum_j x_j^2 + \epsilon}}w_i.
$$

Accumulate the sum of squares in FP32. For widths through 4096, let each thread
load up to four values, retain them through the SIMD/threadgroup reduction, and
write the normalized results without rereading the input row. The accepted
source geometry uses 256 threads and eight SIMD groups; that is source-derived
geometry, not a measured occupancy claim.

Keep the existing fixed-width kernel as the fallback for `D > 4096` and count
both paths in `FastRMSNorm.dispatch_counts`:

- `register_cached` for supported widths;
- `fixed_width_fallback` for larger widths.

The counter proves routing. The readable Python RMSNorm remains the numerical
control.

Complete the first checkpoint:

```bash
pdm run test --week 2 --day 5 -- -k register_cached
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint rmsnorm --model qwen3-0.6b --max-tokens 16
```

Accepted component evidence reduced the RMSNorm operator by **71.36%** at 2K
rows and **83.81%** at 8K rows. These are supplied operator measurements, not
complete-request gains and not fresh results from your checkout.

## Preserve RoPE Positions

RoPE rotates pairs of Q/K coordinates using the absolute incoming token
positions. The fast wrapper must accept one scalar offset, one offset per batch
row, or the model's normalized offset array. Reject an offset vector whose
length disagrees with the batch.

Preserve the existing base, dimension, sequence limit, and traditional-layout
contract. The cache offset is the position of the first incoming token; do not
restart the rotation at zero for every decode call.

```bash
pdm run test --week 2 --day 5 -- -k rope
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint rope --model qwen3-0.6b --max-tokens 16
```

The `rope` checkpoint inherits register-cached RMSNorm. If the fast rotation is
ineligible or fails its tolerance, fall back to the readable RoPE expression
with the same offsets rather than changing cache state.

## Fuse the SwiGLU Elementwise Pass

Qwen's MLP computes:

$$
\operatorname{SwiGLU}(g,u)=\operatorname{SiLU}(g)\odot u,
\qquad \operatorname{SiLU}(g)=g\,\sigma(g).
$$

Fuse the elementwise activation and multiply after the separate packed gate and
up projections. Keep the down projection outside this primitive. Validate equal
shapes and matching supported dtypes, perform the internal arithmetic with the
supplied numerical contract, and return the model-facing dtype.

```bash
pdm run test --week 2 --day 5 -- -k swiglu
pdm run test --week 2 --day 5
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-0.6b --max-tokens 16
```

The whole Day 5 gate checks all three primitive results and proves the feature
sets are cumulative and real. A checkpoint name alone is not enough: the model
must route through the corresponding implementation.

## Measure and Decide at Each Step

For every addition, time the predecessor and candidate with the identical
request. For example:

```bash
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint simd-matmul --model qwen3-0.6b --max-tokens 16
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint rmsnorm --model qwen3-0.6b --max-tokens 16
```

Repeat the pattern for `rmsnorm` versus `rope`, then `rope` versus `swiglu`.
Record the correctness result, path counter where available, coarse product
observation, and fallback. A large component gain can be diluted by projection,
attention, evaluation, and process costs. Keep or reject each primitive on its
own evidence; do not add unrelated percentages.

Continue to [tiled dense prefill attention](./week2-06-operator-lab.md).

{{#include copyright.md}}
