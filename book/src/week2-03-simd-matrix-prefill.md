# 🚧 Week 2 Day 3: SIMD-Matrix Prefill

Day 2 ends with packed W4 projections and a vanilla matrix control. Re-profile
the fixed 128-token prefill and name the dominant category before changing
code. In an older M4 Pro run *after* fused primitives, projections accounted
for 99.1% of attributed prefill time. Treat that as historical context; the
current `quantized-matvec` baseline must be measured on its own.

The `quantized-matvec` checkpoint still uses Day 2's correctness-first
vanilla W4 matrix kernel when the activation has more than eight rows. You will
replace that schedule with a cooperative BF16 SIMD-matrix kernel while preserving the same
quantized-linear interface and the last-row-logits product boundary.

The old checked numbers below are a historical example, not a performance
gate or fresh evidence for this successor. They came from Qwen3-4B on one
20-core M4 Pro running macOS 27 and MLX 0.32.0, with a 128-token prompt, 129
output tokens, two warmups, and two balanced fresh-process samples. Your device
and crossover may differ.

## Establish the Same-Workload Baseline

Start from the checkpoint you already have. Build the extension and run the
focused gate before editing:

```bash
pdm run build-ext
pdm run test --week 2 --day 4
```

Freeze the existing `quantized-matvec` control before the new SIMD matrix
kernel is implemented. Do not select `week2-simd-matmul` yet:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-quantized-matvec --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day3-control.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case quantized-matvec:prefill:128 \
  --warmup 4 --iterations 12 \
  --json-output week2-day3-control-attribution.json
```

Keep these JSON files for the post-edit comparison. Do not compare a new
prefill kernel at one shape with an old result from another shape.

## First Diagnostic: A Partial Tile

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k partial_tiles
```

The first failure should reach the incomplete matrix kernel or its dispatcher.
The witness uses dimensions that do not fill every output tile, so a kernel that
works only on Qwen's aligned main shape is not complete.


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

Retain Day 2's SIMD matvec for `M <= 8`. Route larger activation matrices to
the new tiled kernel and keep the vanilla kernel callable as a bring-up
control. Validate dtype, contiguity, group size, bit width, and matrix
dimensions at the extension boundary before encoding the GPU command. The current dispatch keeps `M <= 8` on
the SIMD matvec and sends `M > 8` to this matrix kernel. Guard partial
output tiles and reduction tails rather than assuming Qwen-only aligned
shapes.

The supplied starter dispatch is `QuantizedMatmul::eval_gpu` in
`src/extensions/src/quantized_matmul.cpp`. Its matrix-shaped Metal entry is
`quantized_matmul_simdgroup_w4a16_g128` in
`src/extensions/src/quantized_matmul.metal`; an equivalent solution may keep
the public dispatch while choosing a different internal kernel name.

The checkpoint feature name is `simd-matmul`. It includes bounded KV capacity and packed W4 projections; the fused
primitives are the next lesson. It does not select a decode-attention branch.

If you want to continue without writing this custom schedule, preserve the
course's `quantized_linear` interface and route the matrix-shaped projection
through `mx.quantized_matmul`. That is a local operator substitution, not a
performance claim and not the separate `--solution mlx` model.

## Task 3: Check Correctness in the Product

Once the new path is connected, get focused feedback before asking the full
model to exercise the checkpoint:

```bash
pdm run build-ext
pdm run test --week 2 --day 4

pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint simd-matmul --model qwen3-0.6b --max-tokens 16
```

An equivalent learner implementation may choose different helper names or a
different correct tiling. The observable contract is the quantized-linear
result, dtype and shape, checkpoint behavior, fallback behavior, and complete
model output—not a private symbol or source-file layout.

## Task 4: Re-profile and Decide

After the SIMD matrix checkpoint passes, measure it alongside the old control
with the same model, phase, token count, prompt rule, prefill-logit mode,
warmups, and iteration count:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-quantized-matvec --variant week2-simd-matmul --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day3-product.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case quantized-matvec:prefill:128 --case simd-matmul:prefill:128 \
  --warmup 4 --iterations 12 \
  --json-output week2-day3-attribution.json
```

Compare these results with the saved pre-edit JSON, then close the loop in
three sentences:

1. which operator category dominated the baseline prefill;
2. whether the candidate changed that category and the matched product phase;
3. what result would make you revert the candidate or test another schedule.

In the older checked run, which started from the `swiglu` checkpoint, the SIMD
schedule reduced attributed projection time by 86.4% and raised fixed-workload
prefill from 106.44 to 721.60 tokens/s. This five-day sequence starts the
matrix lesson from `quantized-matvec`, so those old rows are context rather
than the result of the commands above. They establish no multiplier for your
machine, prompt length, or software version.

Continue to [Day 4 fused model primitives](./week2-04-fused-model-kernels.md).
The old bounded decode-attention and Split-K labs are retained only as
[historical experiments](./appendix-performance.md#retired-week-2-experiments).

{{#include copyright.md}}
