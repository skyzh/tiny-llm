# 🚧 Week 2 Day 7: Split-K Prefill

> **Status: Experimental.** See the
> [Week 2 verification matrix](./week2-overview.md#verification-status) for
> what is continuously tested, locally measured, and still under review.

Day 6's cooperative loads brought long-row prefill near MLX. Its follow-up sweep shows a different
problem at short prefill: Qwen's narrow K/V projections do not launch enough
independent result tiles to occupy the GPU. Today we split the reduction
dimension only until that grid is large enough.

This chapter is not a general split-K library. It optimizes the model shapes we
actually run:

| Model | Reduction `N` | Q output `K` | K/V output `K` |
|---|---:|---:|---:|
| Qwen3-4B | 2,560 | 4,096 | 1,024 |

## Why Split the Reduction Dimension?

For `C = A @ W.T`, Day 6 launches:

```plain
ceil(M / 32) * ceil(K / 32) threadgroups
```

Split-K adds a partition grid dimension:

```plain
partial[p, :, :] = A[:, N_start[p]:N_end[p]]
                    @ W[:, N_start[p]:N_end[p]].T
C = reduce(partial, partition axis)
```

This exposes more independent work, but rereads part of `A`, allocates a
temporary tensor, and launches a reduction kernel. It is useful only while the
original two-dimensional grid is under-filled.

## Task 1: Reproduce the Under-Filled Grid

Task 1 changes no function. Benchmark the existing
`quantized_matmul_simdgroup_w4a16_g128` Day 6 kernel before editing the Split-K
stubs.

Begin with the narrow K projection at `M=32` before changing dispatch. This is
the smallest baseline needed to reproduce the under-filled Day 6 grid; Task 4
runs the full all-projection, all-row sweep after Split-K exists:

```bash
pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section prefill-projections --context 32 --prefill-projection k \
  --warmup 5 --iterations 30
```

Record synchronized Day 6 and MLX latency before implementing Split-K. The
narrow K/V shape is the clearest small-grid case. Large output widths or prompt
lengths may already have enough row-by-column tiles and should become controls.

## Task 2: Reuse the Day 6 Kernel for Each Partition

Implement `quantized_matmul_simdgroup_splitk_w4a16_g128` in
`src/extensions/src/quantized_matmul.metal`, reusing the Day 6 tiled helper
behind `quantized_matmul_simdgroup_w4a16_g128`.

Add `group_id.z` as the partition index. Every partition must:

- have the same reduction length;
- start and end on a 128-value quantization-group boundary;
- reuse the validated Day 6 loader, dequantizer, and 32×32 tile;
- write to its own `[M, K]` plane without atomics.

Store partial planes in BF16 to keep the temporary small and perform the final
sum in FP32 before the output cast. This introduces one extra BF16 rounding
boundary compared with the unsplit FP32 accumulator, so tests use a
BF16-appropriate tolerance. An FP32 temporary is a useful bring-up oracle, but
it doubles the partial-buffer traffic.

## Task 3: Choose Partitions From Occupancy

Modify `QuantizedMatmul::eval_gpu` in
`src/extensions/src/quantized_matmul.cpp` to select the partition count and
dispatch the Split-K kernel. Keep `tiny_llm_ext::quantized_matmul` and its
Python binding unchanged; the existing `use_split_k` argument carries this
cumulative checkpoint.

Use a small explicit policy:

```plain
base_groups = ceil(M / 32) * ceil(K / 32)
split_k = min(16, floor(320 / base_groups), N / 128)
decrease split_k until N is divisible by split_k * 128
use Day 6 unchanged when split_k <= 1
```

For the Qwen3-4B target, use roughly 320 threadgroups
and a cap of 16 as explicit tuning parameters. They are not universal GPU
properties. Unlike a hard-coded prompt-length cutoff, the grid calculation
naturally stops splitting a narrow projection once more row tiles are present,
and stops immediately for already wide grids.

For Qwen3-4B, the policy selects these schedules:

| Projection | Base groups at `M=32` | Selected split at `M=32` | Selected split at `M=128` |
|---|---:|---:|---:|
| Q, `2560 -> 4096` | 128 | 2 | 1 |
| K/V, `2560 -> 1024` | 32 | 10 | 2 |
| O, `4096 -> 2560` | 80 | 4 | 1 |
| MLP gate/up, `2560 -> 9728` | 304 | 1 | 1 |
| MLP down, `9728 -> 2560` | 80 | 4 | 1 |

A split of one means the dispatcher uses the Day 6 kernel unchanged. At the
128-token acceptance shape only the narrow K/V projections remain eligible,
with a two-way split; the other major projections already expose enough output
tiles. At 2,048 tokens every projection uses the unsplit kernel.

Expose the policy through a cumulative `split-k` checkpoint. Keep Day 6
selectable so the benchmark always has an unsplit control.

## Task 4: Reduce and Verify

Implement `quantized_matmul_splitk_reduce` in
`src/extensions/src/quantized_matmul.metal` and complete the corresponding
reduction dispatch in `QuantizedMatmul::eval_gpu`. Do not add a second public
matmul function.

Launch one reduction thread per output element. Sum all partition values in
FP32 and cast once to the model dtype. Test:

- Qwen3-4B's `2560 -> 1024` K/V projection;
- a partial 32-column output tile;
- a shape whose base grid already reaches 320 groups and therefore falls back
  exactly to Day 6.

```bash
pdm run build-ext
pdm run test --week 2 --day 7

for context in 16 32 64 128 2048; do
  for projection in q k v o gate up down; do
    pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
      --section prefill-projections --context "${context}" \
      --prefill-projection "${projection}" --include-split-k
  done
done
```

## Benchmark Analysis: Complete Week 2

Compare Day 6, Day 7, and MLX at short, acceptance, and long prompt lengths.
Split-K should help only while the unsplit output grid is under-filled. Verify
that one-token decode remains unchanged because it still dispatches to Day 3's
matvec, and that sufficiently large prefill shapes select the unsplit Day 6
kernel instead of paying for partial storage and reduction.

Keep a short complete-model control beside the under-filled shape sweep, then
run the fixed Week 2 acceptance workload from Day 3. The
[performance appendix](./appendix-performance.md) is the single place for the
measured hardware, dependency versions, checkpoint table, and final MLX ratios.

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-simd-matmul --variant week2-split-k --variant mlx \
  --model qwen3-4b --input-len 32 --output-len 33 --warmup 2 \
  --prefill-logits last

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-simd-matmul --variant week2-split-k --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-simd-matmul --variant week2-split-k --variant mlx \
  --model qwen3-4b --input-len 2048 --output-len 129 --warmup 2 \
  --prefill-logits last
```

Repeat the operator comparison at the 128-token acceptance shape and at a long
control such as 2,048 tokens. Attach the three end-to-end comparisons and the
per-projection SIMD/Split-K/MLX table at each crossover candidate. Retain
Split-K only below the measured crossover: it must improve the under-filled
projection, preserve one-token decode, and fall back exactly to Day 6 when the
ordinary result grid is already occupied. Record the accumulation and reduction
dispatches beside the calculated partition policy and operator table. The final
stretch-goal acceptance run must still reach 80%
of MLX in both phases.

The [reference checkpoint](./appendix-performance.md#day-7-split-k-only-below-the-crossover)
pairs the short-shape operator gains with the end-to-end result and keeps the
neutral acceptance and long controls separate. Verify directly that the short
shape executes the accumulation and merge pipelines, while the calculated
policy names the partitions and the shape sweep prevents their overhead from
leaking into occupied controls. Week 3 then changes the benchmark itself:
request turnover and dense KV reconstruction, rather than another static
projection, become the measured serving bottleneck.

The Week 2 loop is now complete:

```plain
optimize matvec -> benchmark decode -> optimize model kernels -> benchmark decode
-> optimize attention -> benchmark prefill -> optimize cooperative matmul
-> measure tile occupancy -> optimize split-K -> benchmark the complete checkpoint
```

Week 3 inherits these projection schedules. Paging is evaluated separately on
cache writes, direct page reads, attention time, and end-to-end throughput; it
does not receive credit for the Day 7 projection gain.

{{#include copyright.md}}
