# 🚧 Week 2 Day 7: Split-K Prefill

> 🚧 This chapter is under review and may change.

Day 6 made each 32×32×32 tile efficient. Its follow-up sweep shows a different
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

Benchmark Q, K, V, gate, up, and down projections at `M = 16, 32, 64, 128`
before changing dispatch:

```bash
pdm run bench-week2-operators --model qwen3-4b \
  --context 32 --prefill-projection k --warmup 5 --iterations 30
```

Record synchronized Day 6, requested split-K, and MLX latency. The narrow K/V
shape is the clearest small-grid case. Large output widths or prompt lengths
may already have enough row-by-column tiles and should become controls.

## Task 2: Reuse the Day 6 Kernel for Each Partition

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

Use a small explicit policy:

```plain
base_groups = ceil(M / 32) * ceil(K / 32)
split_k = min(16, floor(320 / base_groups), N / 128)
decrease split_k until N is divisible by split_k * 128
use Day 6 unchanged when split_k <= 1
```

For the Qwen3-4B course checkpoint, use a target of roughly 320 threadgroups
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

Launch one reduction thread per output element. Sum all partition values in
FP32 and cast once to the model dtype. Test:

- Qwen3-4B's `2560 -> 1024` K/V projection;
- a partial 32-column output tile;
- a shape whose base grid already reaches 320 groups and therefore falls back
  exactly to Day 6.

```bash
pdm run build-ext
pdm run test --week 2 --day 7
```

## Complete the Optimization Loop

Compare Day 6, Day 7, and MLX at short, acceptance, and long prompt lengths.
Split-K should help only while the unsplit output grid is under-filled. Verify
that one-token decode remains unchanged because it still dispatches to Day 3's
matvec, and that sufficiently large prefill shapes select the unsplit Day 6
kernel instead of paying for partial storage and reduction.

Run the fixed Week 2 acceptance workload from Day 2 after the shape sweep. The
[performance appendix](./appendix-performance.md) is the single place for the
reference machine, dependency versions, measured checkpoint table, and final
MLX ratios.

The Week 2 loop is now complete:

```plain
optimize matvec -> profile decode -> optimize attention and small kernels
-> profile prefill -> optimize cooperative matmul -> profile tile occupancy
-> optimize split-K -> benchmark the complete checkpoint
```

Week 3 inherits these projection schedules. Paging is evaluated separately on
cache writes, direct page reads, attention time, and end-to-end throughput; it
does not receive credit for the Day 7 projection gain.

{{#include copyright.md}}
