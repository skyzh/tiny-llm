# 🚧 Week 2 Day 6: SIMD-Matrix Prefill

Day 5 leaves one-token decode on the Day 3 matvec and 128-token prefill on the
correctness-first vanilla matrix path for `M > 8`. The Day 6 starter already
contains the cumulative primitive and dispatch, the
`quantized_matmul_simdgroup_w4a16_g128` Metal shell, the course-owned
`CooperativeTileLoader`/`CooperativeBlockMMA` boundary, and the
`logits_to_keep` model switch. You complete those surfaces without changing
the public quantized-linear API.

Implement the matrix path in four cumulative slices: preserve the `M <= 8`
matvec dispatch, add the 32×32×32 tile, make device loads contiguous, reuse one
group's scale/bias across its four reduction tiles, then move the final-logit
slice before the vocabulary projection. The day gate checks aligned and
partial tiles as well as the loader:

```bash
pdm run build-ext
pdm run test --week 2 --day 6
```

After that gate passes, run one matched 128-token prefill and one real-shape
projection control. The live model must use the tiled path; an isolated kernel
result is not enough.

MLX remains an external performance denominator; the SIMD-matrix path in your
solution continues to call the C++/Metal primitive you implement for every
projection.

The implementation remains deliberately narrow:

- W4A16 weights with four bits and group size 128;
- BF16 activations, quantization parameters, and output;
- Qwen3-4B projection dimensions;
- FP32 matrix accumulators;
- the Day 3 SIMD matvec remains in use for `M <= 8`.

## From a Matvec to a Cooperative Tile

The vanilla one-thread dot product and a single-group 8×8 tile are useful
Metal bring-up controls, but neither provides enough cooperative reuse for
multi-row prefill. Compare both with the Python MLX correctness oracle. The
performance schedule must share both activations and dequantized weights across
a larger result tile.

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

Implement the small course-owned boundary in
`src/extensions/src/cooperative_matrix.h`. `CooperativeTileLoader` assigns one
contiguous source chunk to each thread, uses a branch-free full-tile path, and
zero-fills the edge-safe path. `CooperativeBlockMMA` loads and accumulates
direct Metal `simdgroup_matrix` fragments. The required solution does not use
Steel `BlockLoader` or `BlockMMA`.

This helper does not hide the exercise. Your solution still owns the W4A16
unpacking, dequantization, tile layout, direct matrix-fragment bookkeeping,
primitive, dispatch, split policy, and reduction; it does not call MLX's
quantized-matmul operator.

## Task 1: Preserve the Workload Dispatch

Modify `QuantizedMatmul::eval_gpu` in
`src/extensions/src/quantized_matmul.cpp` and
`quantized_matmul_simdgroup_w4a16_g128` in
`src/extensions/src/quantized_matmul.metal`. Keep the Day 3
`quantized_matvec_x4_fast_w4a16_g128` function intact for `M <= 8`.

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

Continue modifying `quantized_matmul_simdgroup_w4a16_g128` in
`src/extensions/src/quantized_matmul.metal` and complete the course-owned
`CooperativeTileLoader` TODO in
`src/extensions/src/cooperative_matrix.h`; do not change the public
`quantized_matmul` binding.

Use a cooperative block loader so adjacent threads and each thread's local
reads form contiguous transactions. This is a requirement of the schedule,
not a cosmetic detail. Benchmark Q, K/V, gate/up, and down projections separately
at their Qwen3-4B dimensions so both wide and narrow output grids are covered.

## Task 3: Hoist Quantization Parameters

Continue modifying `quantized_matmul_simdgroup_w4a16_g128` in
`src/extensions/src/quantized_matmul.metal`. This task changes the tiled
kernel's load/reuse strategy, not its C++ or Python signature.

One scale and bias apply to 128 reduction values. Loading them for every
32-value tile repeats the same device access four times. Have one thread load
the scale and bias for each of the 32 output columns into threadgroup memory,
then let the four weight-unpack threads for that column reuse them for the next
four reduction tiles.

Keep the scale, bias, and unpacked operands in BF16 storage, while the matrix
accumulator remains FP32. Cast once when writing the final model output.

## Task 4: Project Only Required Logits

Modify `Qwen3ModelWeek2.__call__` in `src/tiny_llm/qwen3_week2.py` so
`logits_to_keep=1` slices before the vocabulary projection. Do not add a new
extension function for this model-level optimization.

Generation needs only the final prompt row to produce the first sampled token.
Accept `logits_to_keep=1` and apply the vocabulary projection only to that row.
The benchmark applies the same last-logit workload to MLX, while prompt-scoring
callers can still request every logit row.

## Task 5: Verify, Benchmark, and Name the Next Bottleneck

Task 5 adds no function. Verify the cumulative
`QuantizedMatmul::eval_gpu`/`quantized_matmul_simdgroup_w4a16_g128` path and
the `Qwen3ModelWeek2.__call__` projection boundary from Tasks 1-4.

```bash
pdm run build-ext
pdm run test --week 2 --day 6

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-decode-attention --variant week2-simd-matmul --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-decode-attention --variant week2-simd-matmul --variant mlx \
  --model qwen3-4b --input-len 32 --output-len 33 --warmup 2 \
  --prefill-logits last
```

The comparison should separate two cases. Healthy long-`M` projections show
that the tile itself works; a short, narrow K/V projection with too few 32×32
result tiles exposes the occupancy problem Day 7 addresses. If both cases are
slow, the remaining work is still in this tile rather than in reduction
partitioning.

At long `M`, the two-dimensional tile grid is already large. The checked
2,048-row sweep puts the course SIMD path roughly 7–11% above MLX latency for
the major projections, while the Split-K successor is neutral. That control
does not establish parity with MLX; it establishes that multiplying an already
occupied grid would only add a temporary buffer and another launch.

## Benchmark Analysis: Identify Under-Filled Prefill Shapes

Compare the matrix kernel at both an occupied control shape and the short K/V
shape, then benchmark the latter without enabling Split-K:

```bash
for context in 32 128 2048; do
  for projection in q k v o gate up down; do
    pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
      --section prefill-projections --context "${context}" \
      --prefill-projection "${projection}"
  done
done

```

The dispatch formula gives the unsplit 32-row K projection 32 independent
threadgroups.

Use one short K/V row and one long occupied-grid row to establish the shape
difference. Do not select Split-K merely because projections still occupy most
of prefill.

Use the dispatch calculation and short-shape operator sweep to establish that
the unsplit result grid has too few independent threadgroups. Use the matched
long-shape control to show that Split-K does not help once that grid is
occupied; it may still expose a gap inside each tile, which belongs to Day 6
rather than a larger partition grid. The
[reference checkpoint](./appendix-performance.md#day-6-use-cooperative-loads-for-quantized-prefill)
pairs the prefill result with long and short operator controls and the dispatch
geometry that motivates Split-K. The exact final-main samples and method live
in `benchmark_results/task367-final-main/task367-final-main-benchmark-ledger.md`.

The complete shape campaign and attribution are in the performance appendix.

If you want to continue without implementing the tiled kernel, preserve the
same quantized-linear and `logits_to_keep` interfaces and substitute MLX only
for the matrix-shaped projection. Keep the Day 3 decode matvec and the rest of
the course model intact; `--solution mlx` is a different full-model path.

{{#include copyright.md}}
