# 🚧 Week 2 Day 6: SIMD-Matrix Prefill

> 🚧 This chapter is under review and may change.

Day 5 ends by switching the profile from one-token decode to multi-token
prefill. The measured bottleneck changes with the workload: pointwise kernels
no longer dominate, and the Day 3 matrix-vector schedule does not reuse packed
weights efficiently across prompt rows. Quantized projections are now the
largest cost, so today we build a separate matrix schedule for them.

Do not infer this bottleneck from the number of source lines. Use synchronized
operator timings to compare the course projections with MLX as an external
baseline. The instructor reference also records one diagnostic upper bound: a
temporary model ablation routed only prefill projections through
`mx.quantized_matmul`. At 32 prompt tokens it reached 688.66 prefill tok/s,
while the complete MLX model reached 729.34 tok/s, or 94.4%.

That ablation is not a course checkpoint, student task, or valid solution. It
only estimates how much of the model gap can be recovered by improving
quantized prefill matmul. The required path below continues to call the
course-owned C++/Metal primitive for every projection.

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

Use MLX's low-level Steel `BlockLoader` and `BlockMMA` headers as Metal building
blocks. Those helpers provide cooperative loads and matrix-fragment
bookkeeping. The course still owns the W4A16 unpacking,
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
not a cosmetic detail: at a 2,048-row Qwen3-4B shape, representative course
projections effectively match MLX:

| Projection | Course | MLX |
|---|---:|---:|
| Q, `2560 -> 4096` | 6.66 ms | 6.72 ms |
| K, `2560 -> 1024` | 1.84 ms | 1.85 ms |
| MLP gate, `2560 -> 9728` | 15.49 ms | 15.65 ms |
| MLP down, `9728 -> 2560` | 15.66 ms | 15.76 ms |

These are synchronized operator measurements on the reference M4 Pro. They
show that memory transaction shape is part of the matrix schedule: fragment
arithmetic cannot compensate for scalar, strided tile loads.

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

On the reference M4 Pro, Day 6 reached 605.80 prefill tok/s versus MLX's
727.61 at 32 tokens, or 83.3%. The operator itself is already close to MLX for
large `M`, but small K/V projections launch too few 32×32 result tiles to fill
the GPU. That measured under-filled grid is Day 7's input.

At long `M`, the two-dimensional tile grid is already large. Do not force the
next optimization there: additional reduction partitions would only add a
temporary buffer and another launch.

{{#include copyright.md}}
