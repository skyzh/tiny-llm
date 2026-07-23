# 🚧 Appendix: Performance and Profiling

> 🚧 This appendix is under review and may change.

This appendix records the measurements that determined the course order. The
numbers are not additive promises: after one bottleneck shrinks, every other
operator becomes a larger fraction of model time.

## Benchmark Method

The progression runner launches every checkpoint in a fresh process,
alternates their order, performs complete-request warmups, synchronizes lazy
MLX work inside the timer, and reports the median:

```bash
pdm run bench-week2-progression --offline --repeats 3 \
  --variant week2-simd-matmul --variant week2-split-k --variant mlx \
  --model qwen3-4b --input-len 32 --output-len 33 --warmup 2 \
  --prefill-logits last --json-output week2-32.json

pdm run bench-course-progression --offline --repeats 3 \
  --variant week3 --variant mlx \
  --model qwen3-4b --input-len 2048 --output-len 129 --warmup 2 \
  --prefill-logits all --json-output course-2048.json
```

`--prefill-logits last` is a generation-serving workload: both the course and
MLX project only the last prompt row into vocabulary logits. Use
`--prefill-logits all` for prompt scoring, but never compare the two modes.
Decode throughput excludes the first generated token because that token is
produced by prefill.

MLX's published `mlx_lm.benchmark` table uses a 2,048-token prompt and 128
generated tokens. That makes 2K/128 the primary comparison point, not a
long-context proof. Use a context sweep:

| Point | Purpose |
|---:|---|
| 128 | launch overhead and short interactive requests |
| 2,048 | standard MLX-style acceptance comparison |
| 8,192 | long-context attention and KV-cache stress |
| 16,384 | stress point after the 8K path is healthy |

`llama-bench` commonly uses prompt-processing 512 and token-generation 128 by
default, which is another reminder that benchmark lengths are conventions, not
universal workloads. Always publish the exact prompt and output lengths.

The reference machine below is an Apple M4 Pro with a 20-core GPU and 64 GB of
memory. Unless a row says "smoke", it uses two warmups and the median of three
fresh processes.

## Dependency Upgrade

The project upgraded from MLX 0.29.1 to 0.32.0 and from the mlx-lm 0.28 series
to 0.31.3. A matched Qwen3-4B run showed:

| Context | Metric | MLX 0.29.1 | MLX 0.32.0 | Change |
|---:|---|---:|---:|---:|
| 128 | Prefill tok/s | 825.48 | 828.34 | +0.35% |
| 128 | Decode tok/s | 88.32 | 88.08 | -0.27% |
| 2,048 | Prefill tok/s | 816.73 | 820.85 | +0.50% |
| 2,048 | Decode tok/s | 78.42 | 74.81 | -4.60% |

The upgrade does not explain the old course prefill gap. Its important effect
on the experiment is that the MLX denominator is versioned and remeasured.

## How Quantized Matmul Was Identified

At 32 prompt tokens, replacing only the course prefill projections with
`mx.quantized_matmul` reached 688.66 tok/s while the full MLX model reached
729.34 tok/s: 94.4% of MLX. No other course operator was changed. This ablation
showed that quantized matmul, not Python orchestration or attention, explained
almost all of the prefill gap.

The first tiled kernel was still slow because each thread issued scalar,
strided activation reads. Switching to aligned cooperative block loads and a
32×32×32 tile made large Qwen3-4B projections essentially match MLX:

| Projection at `M=2048` | Course | MLX |
|---|---:|---:|
| Q, `2560 -> 4096` | 6.66 ms | 6.72 ms |
| K, `2560 -> 1024` | 1.84 ms | 1.85 ms |
| MLP gate, `2560 -> 9728` | 15.49 ms | 15.65 ms |
| MLP down, `9728 -> 2560` | 15.66 ms | 15.76 ms |

The next sweep showed under-filled small K/V grids at short `M`. Day 7
therefore splits K only until the result grid approaches 320 threadgroups,
with at most 16 equal, 128-aligned partitions.

The final 32-token median was:

| Checkpoint | Prefill tok/s | Decode tok/s | Prefill / MLX |
|---|---:|---:|---:|
| Day 6 cooperative matmul | 605.80 | 75.95 | 83.3% |
| Day 7 split-K | 671.80 | 75.93 | 92.3% |
| MLX 0.32.0 | 727.61 | 88.48 | 100% |

Split-K adds 10.9% prefill and no decode gain. At large `M`, the base grid is
already full and the dispatch falls back to Day 6.

## Current Qwen3-4B Results

The table separates the completed dense-cache Week 2 checkpoint from Week 3's
direct paged attention path:

| Prompt / decode | Path | Prefill tok/s | Prefill / MLX | Decode tok/s | Decode / MLX |
|---|---|---:|---:|---:|---:|
| 32 / 32 | Week 2 Day 7 | 671.80 | 92.3% | 75.93 | 85.8% |
| 32 / 32 | MLX | 727.61 | 100% | 88.48 | 100% |
| 128 / 32 | Week 2 Day 7 | 623.51 | 74.8% | 73.30 | 82.6% |
| 128 / 32 | MLX | 833.04 | 100% | 88.79 | 100% |
| 2,048 / 128 | Week 2 dense attention | 586.09 | 78.3% | 47.27 | 62.4% |
| 2,048 / 128 | Week 3 paged | 653.96 | 87.4% | 58.88 | 77.7% |
| 2,048 / 128 | MLX | 748.41 | 100% | 75.74 | 100% |
| 8,192 / 128 | Week 3 paged | 427.01 | 75.1% | 31.42 | 56.7% |
| 8,192 / 128 | MLX | 568.74 | 100% | 55.44 | 100% |

The 2K and 8K rows use prompt scoring, one warmup or more, and the median of
three fresh processes. The standard 2K checkpoint is inside the requested
70-80% decode band and above it for prefill. The 8K comparison shows the
remaining boundary clearly: paged FlashAttention prefill still meets the
target, while long-context decode does not.

## Paged Attention: Copies Versus Indirection

Paging does not make an indirect K/V read intrinsically faster than a
contiguous read. Its expected advantages are stable reusable storage, no dense
cache repack, and attention that reads live pages directly. All three must be
present in the benchmark.

The original implementation used Python slice assignment for each page append.
In MLX's functional graph that produced a whole-array update instead of the
small physical write the source suggested. A synchronized append cost about
159 µs per layer. The fixed `paged_cache_update` primitive aliases the existing
page buffer and dispatches a kernel over only the appended slice; geometric
pool growth remains the only full storage copy.

The original decode attention also assigned only `D` threads to a final loop
over 32 partial outputs and reserved about 16.8 KiB of scratch. The current
Qwen D=128 path:

- specializes four contiguous values per lane;
- transposes partials through a compact 32×32 tile;
- uses `simd_sum` for the final reduction;
- reserves about 4.25 KiB of scratch;
- retains a correct generic BF16 fallback for non-Qwen head dimensions.

Qwen3-4B and 8B use four query heads per K/V head. The retained GQA4 decode
specialization computes those four query heads in one threadgroup so each K/V
load is reused four times. It also uses base-2 exponentials in the online and
final softmax reductions. The generic path remains available for other GQA
ratios and head dimensions.

At 2K, the measured progression was:

| Path | Decode tok/s |
|---|---:|
| Original direct paged kernel | 36.17 |
| Paged storage plus dense gather/attention ablation | 46.15 |
| Corrected direct paged kernel before GQA4 reuse | 54.54 |
| GQA4 reuse plus current quantized matvec | 58.88 |
| MLX | 75.74 |

The final direct page path is 24.6% faster than Week 2's dense-cache decode;
prefill is 11.6% faster as well. That is the full-model comparison that
supports the claim that paging removes useful cache work.

For prefill, the same scalar-load mistake found in quantized matmul appeared in
the paged FlashAttention tile. Qwen's 128-token pages make each 32-token K/V
tile physically contiguous, so a cooperative block loader can use aligned
reads while the generic page-crossing fallback remains available. Together
with base-2 online softmax, this raised the retained paired 8K prefill result
from the earlier 384.88 tok/s baseline to 427.01 tok/s. The paired median,
rather than a faster isolated process, is the acceptance result.

## Decode Profile and CLI Tools

`xcrun xctrace` is available from the command line:

```bash
xcrun xctrace list templates

xcrun xctrace record \
  --template "Metal System Trace" \
  --output /tmp/tiny-llm-decode.trace \
  --launch -- pdm run bench --solution tiny_llm_ref --loader week3 \
    --model qwen3-4b --num-seqs 1 \
    --min-input-len 2048 --max-input-len 2048 \
    --min-output-len 33 --max-output-len 33 --warmup 1 \
    --prefill-logits last

xcrun xctrace export --input /tmp/tiny-llm-decode.trace --toc
```

The stock Metal System Trace on the reference Xcode installation exposes
queues and command buffers, but its table of contents reports the Shader
Timeline as disabled. Adding the `Metal GPU Counters` instrument from the CLI
also reports that its selected counter profile is unsupported on this host.
Use a compatible custom Instruments template or an MLX `.gputrace` capture for
shader-level counters. Do not infer ALU or bandwidth saturation from the stock
trace, and do not treat trace wall time as a throughput number.

When the stock trace cannot attribute shaders, use synchronized operator
timings and one-factor full-model ablations. At 2K on the final code:

| One-factor run | Decode tok/s | Change from 58.88 baseline |
|---|---:|---:|
| Replace course quantized matvec with MLX | 60.33 | +2.4% |
| Bypass page-cache writes | 61.14 | +3.8% |
| Bypass paged attention | 88.23 | +49.8% |

These ablations are upper bounds, not alternative correct models, but they
reject both "matvec is the whole decode gap" and "page copies dominate." The
context sweep reinforces the result: increasing context fourfold lowers course
decode by 46.6%, from 58.88 to 31.42 tok/s, while MLX falls 26.8%. Projection
shapes stay fixed; K/V traversal grows with context.

The GQA4 kernel performs roughly four FLOPs per K/V byte before page-table,
softmax, and reduction overhead, so it has low arithmetic intensity even after
four query heads reuse each K/V load. Without supported hardware counters we
cannot claim an exact ALU-versus-bandwidth percentage, but the ablations and
context slope identify K/V access and attention reduction as the next path to
profile—not quantized-matvec arithmetic.

Several plausible changes were measured and removed: reducing the decode
threadgroup to eight SIMD groups, pairing K/V cache writes in one primitive,
and forcing vector K/V loads did not improve end-to-end throughput. A two-pass
attention prototype also regressed because its temporary allocation and second
launch cost more than it saved. The 8K decode row remains below target and is
the next measured optimization boundary.

## Optimization Map

| Measured bottleneck | Retained change | Chapter |
|---|---|---|
| Full-prefix decode recomputation | Dense request KV cache | Week 2 Day 1 |
| Quantized projection weight traffic | Packed W4A16 SIMD matvec, then measured x4/two-packed-word schedule | Week 2 Day 3 |
| Growing short-context attention | Online-softmax decode kernel | Week 2 Day 4 |
| Repeated small graph dispatches | RMSNorm, RoPE, SwiGLU kernels | Week 2 Day 5 |
| Scalar/strided prefill projection loads | Cooperative 32×32×32 quantized matmul | Week 2 Day 6 |
| Under-filled short-prefill result grid | Measured split-K dispatch | Week 2 Day 7 |
| Functional whole-cache page updates | Aliasing page-slice write primitive | Week 3 Day 3 |
| Repeated GQA K/V reads and scalar final reduction | D=128 SIMD reduction plus four-head K/V reuse | Week 3 Day 4 |
| Scalar contiguous-page K/V tile loads | Cooperative paged FlashAttention loads | Week 3 Day 5 |

This is the course progression: optimize one measured cost, benchmark and
profile again, then let the new profile choose the next chapter.

{{#include copyright.md}}
