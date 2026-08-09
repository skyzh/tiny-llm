# 🚧 Week 2 Day 6: SIMD-Matrix Prefill

> **Status: Experimental.** See the
> [Week 2 verification matrix](./week2-overview.md#verification-status) for
> what is continuously tested, locally measured, and still under review.

Day 5 ends by switching the benchmark from one-token decode to multi-token
prefill. Its source trace shows that the 128-token prefill still uses Day 3's
correctness-first vanilla quantized matrix path for `M > 8`. Day 6 replaces that
inherited multi-row schedule, then measures the complete model and the real
projection shapes to decide whether the new path stays.

MLX remains an external performance denominator; the SIMD-matrix path in your
solution continues to call the C++/Metal primitive you implement for every
projection.

> **Optional profiling evidence.** The checked dependency-aware attribution and
> the
> [reference-solution attribution](./appendix-performance.md#checked-operator-attribution-that-selects-each-chapter)
> explain why projections are the reference solution's next target. They are not
> required learner output and do not gate this chapter.

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

Your Metal kernel may use MLX's low-level Steel `BlockLoader` and `BlockMMA`
headers as building blocks. Those helpers provide cooperative loads and
matrix-fragment bookkeeping. Your solution still owns the W4A16 unpacking,
dequantization, tile layout, primitive, dispatch, split policy, and reduction;
it does not call MLX's quantized-matmul operator.

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

Continue modifying `quantized_matmul_simdgroup_w4a16_g128` (and its private
Metal helper, if you factor one) in
`src/extensions/src/quantized_matmul.metal`; do not change the public
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

Inspect the projection sweep as well as complete-model throughput. Continue to
Day 7 when the long-`M` projections are healthy but short, narrow K/V
projections launch too few 32×32 result tiles to fill the GPU. If the same
kernel remains slow at large `M`, improve its loads or matrix schedule before
adding reduction partitions.

At long `M`, the two-dimensional tile grid is already large. Do not force the
next optimization there: additional reduction partitions would only add a
temporary buffer and another launch.

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

Attach the complete-model prefill delta and per-projection tables at 32, 128,
and 2,048 rows. Do not select Split-K merely because projections still occupy
most of prefill. First require the long or wide controls to approach MLX while
the short, narrow projection remains disproportionately slow.

Use the dispatch calculation and short-shape operator sweep to establish that
the unsplit result grid has too few independent threadgroups. Use the matched
long-shape control to rule out costly work inside each tile; if it exposes such
a cost, repair Day 6 before multiplying the grid. The
[reference checkpoint](./appendix-performance.md#day-6-use-cooperative-loads-for-quantized-prefill)
pairs the prefill gain with long and short operator controls and the dispatch
geometry that motivates Split-K. A remaining arithmetic hot spot would send
you back to Day 6 instead.

> **Optional profiling evidence.** A 32/128-row attribution can corroborate the
> shape analysis, but it does not replace the matched complete-model delta,
> projection controls, and dispatch calculation above.

{{#include copyright.md}}
