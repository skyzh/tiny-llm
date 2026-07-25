# 🚧 Week 2 Day 6: SIMD-Matrix Prefill

> 🚧 This chapter is under review and may change.

Day 5 ends by switching the profile from one-token decode to multi-token
prefill. The measured bottleneck changes with the workload: pointwise kernels
no longer dominate, and the Day 3 matrix-vector schedule does not reuse packed
weights efficiently across prompt rows. Quantized projections are now the
largest cost, so today we build a separate matrix schedule for them.

Re-run the dependency-aware kernel profile from Day 2 with
`--case swiglu:prefill:128`. Continue only when projections dominate the
attribution and the complete-model prefill phase moves with their latency. The
[reference-solution profile](./appendix-performance.md#the-kernel-profile-that-selects-each-chapter)
shows that evidence chain. MLX remains an external performance denominator;
the required path in your solution continues to call the C++/Metal primitive
you implement for every projection.

The implementation remains deliberately narrow:

- W4A16 weights with four bits and group size 128;
- BF16 activations, quantization parameters, and output;
- Qwen3-4B projection dimensions;
- FP32 matrix accumulators;
- the Day 3 SIMD matvec remains in use for `M <= 8`.

## From a Matvec to a Cooperative Tile

The vanilla one-thread dot product and a single-group 8×8 tile are useful
correctness oracles, but neither provides enough cooperative reuse for
multi-row prefill. The performance schedule must share both activations and
dequantized weights across a larger result tile.

The optimized kernel assigns four SIMD groups, or 128 threads, to one
32×32×32 tile:

```plain
                  32 output columns
               +--------------------+
32 prompt rows |  four 16x16 SIMD   |
               |  output quadrants  |
               +--------------------+
                         ^
                         |
             shared 32-value K step
```

For each 32-value reduction step, the threadgroup:

1. loads one 32×32 activation tile into padded threadgroup memory;
2. unpacks and dequantizes one 32×32 weight tile there;
3. lets four SIMD groups reuse both tiles;
4. accumulates four 16×16 quadrants from Metal 8×8 matrix fragments;
5. advances to the next reduction tile.

The 40-element shared-memory stride pads the 32-value rows to avoid an
unhelpful bank-access pattern. Tail rows and columns are zero-filled or guarded
at the final store.

Your Metal kernel may use MLX's low-level Steel `BlockLoader` and `BlockMMA`
headers as building blocks. Those helpers provide cooperative loads and
matrix-fragment bookkeeping. Your solution still owns the W4A16 unpacking,
dequantization, tile layout, primitive, dispatch, split policy, and reduction;
it does not call MLX's quantized-matmul operator.

## Task 1: Preserve the Workload Dispatch

Keep the Day 3 decode schedule and add the matrix schedule behind the same
quantized-linear interface:

```plain
M <= 8  -> quantized SIMD matvec
M > 8   -> 32x32x32 quantized SIMD-matrix kernel
```

Expose the new path through the cumulative `simd-matmul` checkpoint. Test the
vanilla, tiled, and MLX results on an aligned shape and on partial row and
column tiles. The result must retain the model-facing 16-bit dtype.

## Task 2: Make Device Loads Contiguous

Use a cooperative block loader so adjacent threads and each thread's local
reads form contiguous transactions. This is a requirement of the schedule,
not a cosmetic detail: fragment arithmetic cannot compensate for scalar,
strided tile loads. Benchmark Q, K/V, gate/up, and down projections separately
at their Qwen3-4B dimensions so both wide and narrow output grids are covered.

## Task 3: Hoist Quantization Parameters

One scale and bias apply to 128 reduction values. Loading them for every
32-value tile repeats the same device access four times. Have one thread load
the scale and bias for each of the 32 output columns into threadgroup memory,
then let the four weight-unpack threads for that column reuse them for the next
four reduction tiles.

Keep the scale, bias, and unpacked operands in BF16 storage, while the matrix
accumulator remains FP32. Cast once when writing the final model output.

## Task 4: Remove Non-Matmul Prefill Waste

Two smaller fixes belong in this checkpoint because the prefill profile shows
them adjacent to the projection work:

- fuse token lookup and W4A16 dequantization in a direct embedding kernel;
- accept `logits_to_keep=1` and apply the vocabulary projection only to the
  final prompt row during generation.

The benchmark applies the same last-logit workload to MLX. Prompt-scoring
callers can still request every logit row.

## Task 5: Verify, Benchmark, and Name the Next Bottleneck

```bash
pdm run build-ext
pdm run test --week 2 --day 6

pdm run bench-week2-progression --offline --repeats 3 \
  --variant week2-simd-matmul --variant mlx \
  --model qwen3-4b --input-len 32 --output-len 33 --warmup 2 \
  --prefill-logits last
```

Inspect the projection sweep as well as complete-model throughput. Continue to
Day 7 when the long-`M` projections are healthy but short, narrow K/V
projections launch too few 32×32 result tiles to fill the GPU. If the same
kernel remains slow at large `M`, improve its loads or matrix schedule before
adding reduction partitions.

At long `M`, the two-dimensional tile grid is already large. Do not force the
next optimization there: additional reduction partitions would only add a
temporary buffer and another launch.

{{#include copyright.md}}
