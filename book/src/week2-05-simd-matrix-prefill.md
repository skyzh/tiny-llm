# 🚧 Matrix Prefill: Reuse a Weight Tile

Your `quantized-matvec` checkpoint keeps weights packed and makes the
small-row projection runnable. A prompt supplies many activation rows, however.
Computing each dot product independently can reload the same weight values for
neighboring rows. You will change the matrix schedule while preserving
`C = A Wᵀ`, the packed representation, and the public quantized-linear API.

Start with the matrix operator gate:

```bash
pdm run build-ext
pdm run test --week 2 --day 5 -- -k task_2
```

These tests diagnose the new tile path and its tails. Failures are expected
before you complete the learner-owned matrix kernel and cooperative loader;
the whole model checkpoint comes later.

## Give the Tile a Useful Job

The work stays in the existing starter files:

```text
src/extensions/src/cooperative_matrix.h
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

Build a 32×32 output tile from 8×8 SIMD-group matrix fragments. For each
32-value slice of the reduction dimension, load an activation tile, unpack the
corresponding W4 weight values, multiply BF16 fragments with FP32 accumulation,
and advance. The same loaded values can serve several output elements before
you fetch the next reduction slice.

This schedule makes synchronization part of correctness. A consumer must not
read shared tile memory before its producers finish, and a producer must not
overwrite a tile another group still uses. Put barriers at those actual data
dependencies. Keep the activation and weight strides explicit; a transposed
weight view does not have the activation tile's indexing.

Rows and columns near an edge need the same care as the center of a matrix.
Zero-fill invalid tile loads and store only valid output elements. For example,
a matrix with 10 rows cannot safely read 32 rows merely because the allocation
following it happens to be accessible. The supplied loader witness places
nonzero data outside the logical rows, so an unmasked load cannot hide behind
zero padding. Partial tiles still need FP32 accumulation.

## Dispatch by Shape

Complete the matrix branch of `QuantizedMatmul::eval_gpu` and
`quantized_matmul_simdgroup_w4a16_g128`. Retain the decode matvec for `M <= 8`;
when matrix dispatch is enabled, larger row counts use the tiled schedule.
Keep `quantized_matmul_vanilla` callable as an arithmetic control.

`M` is the flattened activation-row count, not the attention context length.
Attention's `L` query rows and `S` cached source rows will matter in the final
chapter. Here, row count, reduction width, tile dimensions, and tails are
substeps of implementing one matrix operator, rather than separate model
optimizations.

The `simd-matmul` checkpoint adds only matrix dispatch to packed-W4 projections.
RMSNorm, RoPE, and SwiGLU remain readable. Their custom implementations are not
prerequisites for this checkpoint.

## Complete and Compare the Product

After the tile tests pass, wire the model's quantized weights to the checkpoint
feature and run the complete gate:

```bash
pdm run build-ext
pdm run test --week 2 --day 5
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint simd-matmul --model qwen3-0.6b
```

The gate includes a model witness that makes the later custom primitives fail
if called: reaching this checkpoint must not depend on their future work. It
also checks matrix results, boundary shapes, and the model's output contract.

Use the incoming `quantized-matvec` checkpoint as the same-model control:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-quantized-matvec --variant week2-simd-matmul \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-matrix.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case quantized-matvec:prefill:128 --case simd-matmul:prefill:128 \
  --warmup 4 --iterations 12 --json-output week2-matrix-attribution.json
```

The pair differs in matrix scheduling, while both use packed weights and
readable primitives. Inspect prefill and decode separately. Then try another
prompt length with both controls changed together; do not compare a short
baseline with a long candidate. A tile that helps one shape may leave another
unchanged or make it slower.

Record the result without inferring occupancy from a timing change. The next
[primitive lab](./week2-04-fused-model-kernels.md) begins at `simd-matmul` and keeps
this matrix path as it adds RMSNorm, RoPE, and SwiGLU one at a time.

{{#include copyright.md}}
