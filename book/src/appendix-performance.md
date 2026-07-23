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
  --variant week2-split-k --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-128.json

pdm run bench-serving-progression --offline --repeats 3 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 \
  --prefill-step 128 --json-output serving-qwen3-4b.json
```

`--prefill-logits last` is a generation-serving workload: both the course and
MLX project only the last prompt row into vocabulary logits. Use
`--prefill-logits all` for prompt scoring, but never compare the two modes.
Decode throughput excludes the first generated token because that token is
produced by prefill.

MLX's published `mlx_lm.benchmark` table uses a 2,048-token prompt and 128
generated tokens. That makes 2K/128 a useful static-library comparison point,
not a paging acceptance test or a long-context proof. Use a context sweep:

| Point | Purpose |
|---:|---|
| 128 | fixed Week 2 acceptance and short interactive requests |
| 2,048 | standard MLX-style static stress comparison |
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

## Week 2 Acceptance Result

Week 2 has a fixed single-request acceptance shape: Qwen3-4B, a 128-token
prompt, 128 timed decode steps, last-row logits, two complete warmups, and the
median of three alternating fresh processes. The output length is 129 because
the first generated token belongs to prefill.

| Path | Prefill tok/s | Prefill / MLX | Decode tok/s | Decode / MLX |
|---|---:|---:|---:|---:|
| Week 2 Day 7 | 791.63 | 95.6% | 77.45 | 88.4% |
| MLX 0.32.0 | 828.27 | 100% | 87.62 | 100% |

Both sides clear the 80% course target. Longer static context sweeps remain
useful diagnostics, especially for showing attention's linear growth, but they
do not test the memory-management reasons for paging.

## Week 3 Serving Result

Paging adds an indirect K/V read and is not expected to beat contiguous
attention in an isolated static request. Its acceptance workload must include
requests entering and leaving the batch, incremental growth without a known
cache capacity, dense batch reconstruction, and page reuse.

The serving runner launches dense and paged variants in fresh alternating
processes. A complete warmup compiles the kernels; immediately afterward it
synchronizes and resets every page pool. The measured paged run therefore
starts with zero pages and zero backing capacity rather than inheriting the
warmup allocation.

```bash
pdm run bench-serving-progression --offline --repeats 3 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 \
  --prefill-step 128 --warmup 1 \
  --json-output serving-qwen3-4b.json
```

On the same M4 Pro, the median result was:

| Storage path | Output tok/s | Decode tok/s | Requests/s | Peak KV MiB | Avoidable KV copy MiB |
|---|---:|---:|---:|---:|---:|
| Week 2 dense KV | 35.87 | 58.65 | 0.48 | 1,096 | 209,532 |
| Week 3 paged KV | 46.70 | 104.19 | 0.62 | 576 | 504 |

The paged path improves output and request throughput by 30.2%, aggregate
decode throughput by 77.7%, and reduces peak KV storage by 47.4%. Its measured
avoidable copy volume is 99.8% lower.

The checked-in result
`benchmark_results/m4-pro-qwen3-4b-mlx-0.32.0.json` contains the three samples,
medians, configurations, and host metadata used by the Week 2 and serving
tables.

The copy counters report logical operation volume, not a hardware DRAM
counter. Dense volume includes old K/V copied during each request-cache growth
and live K/V copied into a newly padded batch tensor at every decode step.
Paged volume includes old physical pages copied only when a layer's geometric
pool grows. Appending a token writes just its page slice, and later requests
reuse freed pages.

The paged median reached 1,116 live pages out of 1,152 reserved pages, reused
2,196 page allocations, and recorded 15,840 unused tail slots across layer
caches. It grew the layer pools 144 times because the measured run started
empty. These allocator counters make memory reuse and fragmentation visible;
a static single-request latency table cannot.

This workload validates continuous batching, chunked prefill, incremental
growth, and page reuse. It does not validate prefix sharing or speculative
decoding; those require separate request traces with shared prefixes or cache
rewind events.

## Retained and Removed Optimizations

The course keeps changes with a clear measured role: the x4 quantized matvec,
cooperative prefill matmul, split-K for under-filled grids, page-slice writes,
the compact D=128 paged reduction, and cooperative paged prefill loads.

Two shape-specific duplicates were removed. A pointer-streaming matvec copied
the complete x4 reduction loop for a small end-to-end gain. A four-query-head
paged-decode kernel copied the complete attention recurrence to improve static
latency, even though static latency is no longer Week 3's acceptance metric.
The compact generic Qwen path is easier to teach and still wins decisively in
the serving workload because it removes dense growth and repacking.

Several other plausible changes had already measured poorly and remain
removed: reducing the decode threadgroup to eight SIMD groups, pairing K/V
cache writes in one primitive, forcing vector K/V loads, and a two-pass
attention kernel. Each added work or synchronization without an end-to-end
win.

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

The stock Metal System Trace exposes queues and command buffers on the
reference machine, but its Shader Timeline is disabled and the selected Metal
GPU counter profile is unsupported. Use a compatible custom Instruments
template or an MLX `.gputrace` capture for shader-level counters. Never infer
ALU or bandwidth saturation from an unavailable counter, and do not use trace
wall time as a throughput result.

## Optimization Map

| Measured bottleneck | Retained change | Chapter |
|---|---|---|
| Full-prefix decode recomputation | Dense request KV cache | Week 2 Day 1 |
| Quantized projection weight traffic | Packed W4A16 x4 SIMD matvec | Week 2 Day 3 |
| Growing short-context attention | Online-softmax decode kernel | Week 2 Day 4 |
| Repeated small graph dispatches | RMSNorm, RoPE, SwiGLU kernels | Week 2 Day 5 |
| Scalar/strided prefill projection loads | Cooperative 32×32×32 quantized matmul | Week 2 Day 6 |
| Under-filled short-prefill result grid | Measured split-K dispatch | Week 2 Day 7 |
| Functional whole-cache page updates | Aliasing page-slice write primitive | Week 3 Day 3 |
| Scalar paged final reduction | Compact D=128 SIMD reduction | Week 3 Day 4 |
| Scalar contiguous-page K/V tile loads | Cooperative paged FlashAttention loads | Week 3 Day 5 |

This is the course progression: optimize one measured cost, benchmark and
profile again, then let the new profile choose the next chapter.

{{#include copyright.md}}
