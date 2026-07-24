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
| Qwen3-8B | 4,096 | 4,096 | 1,024 |

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

The reference stores partial planes in BF16 to keep the temporary small and
performs the final sum in FP32 before the output cast. This introduces one
extra BF16 rounding boundary compared with the unsplit FP32 accumulator, so
tests use a BF16-appropriate tolerance. An FP32 temporary is a valid bring-up
oracle, but it did not justify doubling temporary traffic in the measured
course path.

## Task 3: Choose Partitions From Occupancy

Use a small explicit policy:

```plain
base_groups = ceil(M / 32) * ceil(K / 32)
split_k = min(16, floor(320 / base_groups), N / 128)
decrease split_k until N is divisible by split_k * 128
use Day 6 unchanged when split_k <= 1
```

The target of roughly 320 threadgroups and cap of 16 are measured constants
for the reference M4 Pro, not universal GPU properties. Unlike a hard-coded
prompt-length cutoff, the grid calculation naturally stops splitting a narrow
projection once more row tiles are present, and stops immediately for already
wide grids.

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

## Reference Split-K Measurement

This matched run used Qwen3-4B, MLX 0.32.0, mlx-lm 0.31.3, an M4 Pro with a
20-core GPU and 64 GB memory, last-logit prefill, 32 prompt tokens, 33 output
tokens, two complete warmups, and the median of three fresh processes:

| Checkpoint | Prefill tok/s | Decode tok/s | Prefill vs MLX |
|---|---:|---:|---:|
| Day 6 SIMD matrix | 605.80 | 75.95 | 83.3% |
| Day 7 split-K | 671.80 | 75.93 | 92.3% |
| MLX | 727.61 | 88.48 | 100% |

Split-K improves prefill by 10.9% and leaves decode unchanged, as it should:
one-token decode still uses Day 3's matvec. At 2,048 prompt tokens the base
tile grid is already large, so Day 7 falls back to Day 6 and the completed
Week 2 model reaches about 78% of MLX prefill in the current prompt-scoring
campaign.

The final Week 2 acceptance run uses the fixed 128-token prompt and 128-token
decode workload from Day 2. The three-process median is 792.18 prefill tok/s
and 77.41 decode tok/s. MLX 0.32.0 reaches 827.74 and 87.58 tok/s respectively,
so the course path finishes at 95.7% prefill and 88.4% decode.

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
