# 🚧 Week 2 Day 5: SIMD-Matrix Prefill

Day 4 ends with a decision, not a predetermined kernel. Re-profile the fixed
128-token prefill and name the dominant category before changing code. On the
checked M4 Pro run, projections accounted for 99.1% of attributed prefill time.
That observation selects the matrix-shaped projection path for this chapter.

The `swiglu` checkpoint still uses Day 3's correctness-first vanilla W4 matrix
kernel when the activation has more than eight rows. You will replace that
schedule with a cooperative BF16 SIMD-matrix kernel while preserving the same
quantized-linear interface and the last-row-logits product boundary.

The checked numbers in this chapter are one example, not a performance gate.
They come from Qwen3-4B on one 20-core M4 Pro running macOS 27 and MLX 0.32.0,
with a 128-token prompt, 129 output tokens, two warmups, and two balanced
fresh-process samples. Your device and crossover may differ.

## Establish the Same-Workload Baseline

First verify the inherited checkpoint:

```bash
pdm run build-ext
pdm run test --week 2 --day 5
```

Then record the product and attribution baselines that you will repeat after
the change:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-swiglu --variant week2-simd-matmul --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day5-product.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case swiglu:prefill:128 --case simd-matmul:prefill:128 \
  --warmup 4 --iterations 12 \
  --json-output week2-day5-attribution.json
```

Keep the model, phase, token count, prompt rule, prefill-logit mode, warmups,
and iteration count identical across the two checkpoints. Do not compare a
new prefill kernel at one shape with an old result from another shape.

## Task 1: Load One Quantized Tile Cooperatively

Work in the existing quantized-matmul extension surface:

```plain
src/extensions/src/cooperative_matrix.h
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

The output remains

$$
C = A W^T,
$$

where `A` is BF16 and `W` is stored as packed W4 codes with one scale and bias
per group of 128 values. The mathematical operation does not change. Only the
matrix-shaped schedule changes.

Build a 32×32 output tile from 8×8 `simdgroup_matrix` fragments. SIMD groups
cooperate on one 32-value slice of the reduction dimension at a time:

1. load a contiguous activation tile;
2. unpack the matching W4 codes and apply their scale and bias;
3. multiply the BF16 fragments while accumulating in FP32;
4. advance through the reduction dimension;
5. store only in-bounds output elements.

Keep the loader and fragment bookkeeping explicit. The course path does not
call an MLX or Steel quantized-matmul implementation in place of this exercise.
The existing Python equation remains the correctness oracle.

## Task 2: Dispatch by Activation Shape

Retain Day 3's SIMD matvec for `M <= 8`. Route larger activation matrices to
the new tiled kernel and keep the vanilla kernel callable as a bring-up
control. Validate dtype, contiguity, group size, bit width, and matrix
dimensions at the extension boundary before encoding the GPU command.

The supplied starter dispatch is `QuantizedMatmul::eval_gpu` in
`src/extensions/src/quantized_matmul.cpp`. Its matrix-shaped Metal entry is
`quantized_matmul_simdgroup_w4a16_g128` in
`src/extensions/src/quantized_matmul.metal`; an equivalent solution may keep
the public dispatch while choosing a different internal kernel name.

The checkpoint feature name is `simd-matmul`. It includes packed W4
projections and the three fused Day 4 operators. It does not include the
optional decode-attention branch from Day 6.

If you want to continue without writing this custom schedule, preserve the
course's `quantized_linear` interface and route the matrix-shaped projection
through `mx.quantized_matmul`. That is a local operator substitution, not a
performance claim and not the separate `--solution mlx` model.

## Task 3: Check Correctness in the Product

Run the focused day gate before drawing a timing conclusion:

```bash
pdm run build-ext
pdm run test --week 2 --day 5

pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint simd-matmul --model qwen3-4b
```

An equivalent learner implementation may choose different helper names or a
different correct tiling. The observable contract is the quantized-linear
result, dtype and shape, checkpoint behavior, fallback behavior, and complete
model output—not a private symbol or source-file layout.

## Task 4: Re-profile and Decide

Rerun the exact commands from the baseline section. Record three sentences:

1. which operator category dominated the baseline prefill;
2. whether the candidate changed that category and the matched product phase;
3. what result would make you revert the candidate or test another schedule.

In the checked run, the SIMD schedule reduced attributed projection time by
86.4% and raised fixed-workload prefill from 106.44 to 721.60 tokens/s. Those
large effects justify keeping it for that source tree and workload. They do
not establish the same multiplier on another model, Apple GPU, prompt length,
or software version.

Day 6 is an optional workload-conditioned operator lab. You may take that
branch to study bounded decode attention, or continue directly to Day 7. Day
7 starts from this `simd-matmul` checkpoint either way.

{{#include copyright.md}}
