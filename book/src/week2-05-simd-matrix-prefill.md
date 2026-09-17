# 🚧 Week 2 Day 5: SIMD-Matrix Prefill

Day 4 ends with a decision, not a predetermined kernel. Re-profile prefill and
separate QKV/output attention projections from MLP gate/up/down projections
before changing code. The bounded Day 5 question is whether the packed-W4
projection path needs a matrix-shaped schedule once the activation has more
than eight rows.

The `swiglu` checkpoint still uses Day 3's correctness-first vanilla W4 matrix
kernel when the activation has more than eight rows. You will replace that
schedule with a cooperative BF16 SIMD-matrix kernel while preserving the same
quantized-linear interface and the last-row-logits product boundary.

Use the full 128/512/2K/8K/32,640 prompt matrix from the Week 2 overview for
product evidence. A 128-token prompt remains a useful matched regression
control, not the whole product story. Your device and crossover may differ.

## Establish the Same-Workload Baseline

Start from the checkpoint you already have. Build the extension and run the
focused gate before editing:

```bash
pdm run build-ext
pdm run test --week 2 --day 5
```

Freeze both baselines next. You will repeat these exact commands after the
kernel change:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-swiglu --variant week2-simd-matmul --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 128 --warmup 2 \
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

Open the three existing extension files; this task stays inside that surface:

```plain
src/extensions/src/cooperative_matrix.h
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

Keep the operation fixed while you change its schedule:

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
projections and the three fused Day 4 operators. It is the unchanged control
for both replacement chapters and includes neither Day 6 nor Day 7 candidate.

If you want to continue without writing this custom schedule, preserve the
course's `quantized_linear` interface and route the matrix-shaped projection
through `mx.quantized_matmul`. That is a local operator substitution, not a
performance claim and not the separate `--solution mlx` model.

## Task 3: Check Correctness in the Product

Once the new path is connected, get focused feedback before asking the full
model to exercise the checkpoint:

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

Rerun the exact commands from the baseline section, then close the loop in three
sentences:

1. which model component dominated the baseline prefill;
2. whether the candidate changed that category and the matched product phase;
3. what result would make you revert the candidate or test another schedule.

Retain the schedule only when its targeted phase improves and the full-request
matrix does not reveal a contradictory regression. One prompt length cannot
establish the same crossover on another model, Apple GPU, or software version.

Day 6 is an optional long-context dense-attention experiment. You may take that
branch or continue directly to Day 7's fused gate+up and SwiGLU candidate. Day
7 starts from this `simd-matmul` checkpoint either way.

{{#include copyright.md}}
