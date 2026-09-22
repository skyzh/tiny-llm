# 🚧 Week 2 Day 4: SIMD Matrix Prefill

The `quantized-matvec` checkpoint has two schedules for the same packed-W4
equation:

$$C = A W^T.$$

Small row counts use the SIMD matvec. Larger prefill matrices still use the
readable one-thread-per-output Metal control. Day 4 keeps the packed weights and
changes only that matrix-shaped schedule.

Your seam is the SIMD-group matrix kernel and `QuantizedMatmul::eval_gpu`
dispatch in:

```text
src/extensions/src/cooperative_matrix.h
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

## First Diagnostic: A Partial Tile

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k partial_tiles
```

The first failure should reach the incomplete matrix kernel or its dispatcher.
The witness uses dimensions that do not fill every output tile, so a kernel that
works only on Qwen's aligned main shape is not complete.

## Reuse Activations and Packed Weights

Flatten all leading activation dimensions into `M`. The logical weight shape is
`N, K`; eight W4 codes occupy one `uint32`, with BF16 scale and bias per group of
128 logical weights.

Build a 32×32 output tile from 8×8 `simdgroup_matrix` fragments. Cooperating
SIMD groups repeatedly:

1. load a contiguous activation tile;
2. unpack the matching W4 codes and apply their scale and bias;
3. multiply BF16 fragments while accumulating in FP32;
4. advance through the K dimension;
5. store only in-bounds output elements.

The mathematical output is unchanged from Day 3. The readable vanilla kernel
is the arithmetic fallback; the Day 3 matvec remains the small-row path.

## Dispatch by Row Shape

Keep dispatch inspectable:

| Shape | Schedule | Reason |
|---|---|---|
| `M <= 8` | Day 3 SIMD matvec | Avoid mostly empty matrix tiles during decode |
| `M > 8` | Day 4 SIMD-group matrix kernel | Reuse tiles across prefill rows |
| Unsupported dtype/layout/quantization | reject before encoding | Do not reinterpret an invalid buffer |

Validate ranks, contiguity, K agreement, output shape, BF16 activation and
metadata dtypes, W4 bit width, and group size 128 at the C++ boundary. Guard
edge rows, columns, and the reduction tail before load or store.

The starter's cumulative feature map already makes `simd-matmul` inherit
bounded capacity and packed weights. It must not require the later RMSNorm,
RoPE, SwiGLU, or attention implementations.

## Complete the `simd-matmul` Checkpoint

```bash
pdm run build-ext
pdm run test --week 2 --day 4
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint simd-matmul --model qwen3-0.6b --max-tokens 16
```

The whole gate checks the checkpoint and a partial-tile GPU result against the
readable control. If the tiled schedule is not valid, keep the vanilla matrix
kernel while preserving the same `quantized_linear` interface.

## Measure and Decide

Use the incoming checkpoint as the same-model control:

```bash
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-0.6b --max-tokens 16
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint simd-matmul --model qwen3-0.6b --max-tokens 16
```

Record the exact request, whether the GPU comparison and complete model passed,
and whether the product observation is stable enough to interpret. A faster
isolated tile is component evidence; it does not prove the whole request is
faster. Do not infer occupancy, bandwidth, or cache behavior from elapsed time
alone.

Keep the schedule when it preserves the packed-W4 equation across aligned and
tail shapes and the matched product does not regress. Revert matrix dispatch to
the readable kernel if correctness, tail handling, or the product control fails.

Continue to the [RMSNorm, RoPE, and SwiGLU lab](./week2-05-simd-matrix-prefill.md).

{{#include copyright.md}}
