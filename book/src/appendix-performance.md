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
memory. Static Week 2 rows use two complete warmups; the continuous-serving
rows use one. Both report the median of three fresh alternating processes.

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

## Week 2 Performance by Chapter

Week 2 has one fixed acceptance shape: Qwen3-4B, a 128-token prompt, 128 timed
decode steps, last-row logits, two complete warmups, and the median of three
alternating fresh processes. The output length is 129 because prefill produces
the first generated token.

Each row is cumulative. Day 2 deliberately retains the Day 1 checkpoint while
it establishes the synchronized benchmark and profile that choose Day 3.

| Chapter | Cumulative checkpoint | Prefill tok/s | Decode tok/s | Output tok/s | Change selected by the preceding profile |
|---|---|---:|---:|---:|---|
| Day 1 | Dense request KV cache | 726.25 | 24.51 | 23.89 | Stop full-prefix decode recomputation. |
| Day 2 | Benchmark and profile | 726.25 | 24.51 | 23.89 | Measure packed projection weight traffic. |
| Day 3 | Quantized matvec | 104.82 | 59.17 | 38.10 | Keep weights packed and add the x4 decode kernel. |
| Day 4 | Decode attention | 104.51 | 60.23 | 38.50 | Replace growing score/softmax/value work. |
| Day 5 | RMSNorm, RoPE, and SwiGLU | 106.03 | 77.09 | 44.98 | Remove the newly exposed repeated graph launches. |
| Day 6 | SIMD-matrix prefill | 793.15 | 76.98 | 70.69 | Fix the quantized matrix path exposed by Day 3. |
| Day 7 | Split-K prefill | 792.18 | 77.41 | 71.05 | Fill the GPU only for under-occupied short projections. |
| Baseline | MLX 0.32.0 | 827.74 | 87.58 | 79.81 | External denominator. |

### Day 1: Cache the Prefix

The dense cache makes prefill a one-time cost, but every decode projection
still reads dense weights. Day 1 therefore starts with respectable prefill and
only 24.51 decode tok/s. The result gives Day 2 a real cached baseline to
profile.

### Day 2: Measure Before Optimizing

Day 2 changes the measurement discipline rather than the model. Synchronized
operator timings and a Metal trace identify projection weight reads as the
largest actionable decode cost. That evidence selects packed quantized matvec
for Day 3; the CLI profiling workflow is recorded later in this appendix.

### Day 3: Keep Weights Packed

The x4 W4A16 matvec raises decode from 24.51 to 59.17 tok/s, a 141.4% gain.
Prefill intentionally falls from 726.25 to 104.82 tok/s because the packed
checkpoint still sends matrix-shaped inputs through the readable one-thread
quantized kernel. This is not hidden as a temporary implementation detail: the
new prefill profile makes quantized matrix multiplication the next dominant
cost.

### Day 4: Follow Attention as Context Grows

At the fixed short context, online-softmax decode attention raises decode by
only 1.8%. Context sweeps justify retaining it because its share grows with the
cache length. The small acceptance gain also prevents the course from
incorrectly presenting attention as the main short-context bottleneck.

### Day 5: Optimize the Newly Exposed Launches

Day 5 applies three independently measurable changes. Fast RMSNorm raises
decode by 11.1%, fast RoPE adds 8.2%, and fused SwiGLU adds another 6.5% at
their cumulative checkpoints. The completed day reaches 77.09 decode tok/s;
the profile then shifts decisively to prefill matrix multiplication.

### Day 6: Repair Quantized Prefill

At 32 prompt tokens, replacing only the course projections with
`mx.quantized_matmul` reached 688.66 tok/s while the full MLX model reached
729.34 tok/s. No other operator changed, so the ablation isolated quantized
matmul rather than Python orchestration or attention.

The first tiled course kernel remained slow because each thread issued scalar,
strided activation reads. Aligned cooperative block loads and a 32×32×32 tile
made the large Qwen3-4B projections essentially match MLX:

| Projection at `M=2048` | Course | MLX |
|---|---:|---:|
| Q, `2560 -> 4096` | 6.66 ms | 6.72 ms |
| K, `2560 -> 1024` | 1.84 ms | 1.85 ms |
| MLP gate, `2560 -> 9728` | 15.49 ms | 15.65 ms |
| MLP down, `9728 -> 2560` | 15.66 ms | 15.76 ms |

That repair raises prefill from 106.03 to 793.15 tok/s, a 648.0% gain, while
leaving vector decode unchanged.

### Day 7: Split K Only Below the Crossover

The Day 6 shape sweep finds under-filled small K/V grids only at short `M`.
Day 7 splits K until the result grid approaches 320 threadgroups, using at most
16 equal, 128-aligned partitions. At the 32-token control point:

| Checkpoint | Prefill tok/s | Decode tok/s | Prefill / MLX |
|---|---:|---:|---:|
| Day 6 cooperative matmul | 602.25 | 82.09 | 82.8% |
| Day 7 split-K | 668.66 | 82.04 | 91.9% |
| MLX 0.32.0 | 727.28 | 88.34 | 100% |

Split-K adds 11.0% prefill at this short shape and no decode gain. At the
128-token acceptance shape the base grid is already occupied, so Day 7 falls
back to Day 6 and remains neutral.

The completed Week 2 path reaches 95.7% of MLX prefill, 88.4% of MLX decode,
and 89.0% of MLX end-to-end output throughput. All three exceed the 80% course
target. Longer static sweeps remain useful attention diagnostics, but they do
not test the memory-management reasons for paging.

## Week 3 Performance by Chapter

Paging adds indirect K/V reads and is not expected to beat contiguous
attention for one preallocated static request. Week 3 therefore measures a
serving workload with request turnover, incremental unknown-size growth,
chunked admission, dense batch reconstruction, and page reuse:

```bash
pdm run bench-serving-progression --offline --repeats 3 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 \
  --prefill-step 128 --warmup 1 \
  --json-output serving-qwen3-4b.json
```

A complete warmup compiles the kernels. The runner then synchronizes and resets
every page pool, so the measured paged run starts with zero pages and zero
backing capacity.

| Chapter | Measured checkpoint | Primary result | Change from the preceding comparable path |
|---|---|---|---|
| Day 1 | Continuous scheduler | Defines request turnover and active-batch throughput. | Establishes the serving workload. |
| Day 2 | Chunked admission with dense reconstruction | 35.87 output tok/s; 58.65 decode tok/s | Establishes the dense serving baseline. |
| Day 3 | Paged storage with compatibility gather | 41.60 output tok/s; 79.19 decode tok/s | +16.0% output; +35.0% decode; -50.6% copy volume. |
| Day 4 | Direct paged decode | 104.19 aggregate decode tok/s | +31.6% decode over the compatibility gather path. |
| Day 5 | Paged FlashAttention at 8K | 427.01 prefill tok/s | +10.9% over the earlier page-walking prefill. |
| Performance lab | Complete direct paged path | 46.70 output tok/s; 0.62 requests/s | +30.2% output and request throughput over dense serving. |

Day 1 introduces scheduling, not a kernel speedup. Day 2 makes the hidden cost
measurable: appending one token still reconstructs a padded dense batch. Day 3
makes pages canonical but retains `gather_dense()` as a compatibility
checkpoint. Days 4 and 5 then remove that compatibility movement for decode
and long-query prefill respectively.

The cumulative serving endpoints on the same workload are:

| Storage and attention path | Output tok/s | Decode tok/s | Requests/s | Peak KV MiB | Avoidable KV copy MiB |
|---|---:|---:|---:|---:|---:|
| Dense growth and reconstruction | 35.87 | 58.65 | 0.478 | 1,096 | 209,532 |
| Paged storage plus dense gather | 41.60 | 79.19 | 0.554 | — | 103,445 |
| Direct paged attention | 46.70 | 104.19 | 0.622 | 576 | 504 |

The compatibility row omits peak storage because its current counter takes the
maximum of the page pool and dense staging allocation instead of their sum.
Reporting that lower bound as an exact peak would be misleading.

Direct paged attention improves output and request throughput by 30.2%,
aggregate decode by 77.7%, and peak KV storage by 47.4% relative to dense
serving. Avoidable logical copy volume falls by 99.8%. Relative to paged
storage plus gather, the direct operator adds 12.3% output throughput, 31.6%
decode throughput, and removes 99.5% of the remaining copy volume.

Day 5 is evaluated separately at the shape where query tiling matters. On the
8K Qwen3-4B run, paged FlashAttention raises prefill from 384.88 to 427.01
tok/s. MLX reaches 568.74 tok/s, so the page-aware course path reaches 75.1%.
One-token decode continues to dispatch to the Day 4 vector schedule.

The checked-in result
`benchmark_results/m4-pro-qwen3-4b-mlx-0.32.0.json` contains the published
acceptance and direct-serving samples, medians, configurations, and host
metadata. Chapter checkpoint rows use the same fresh-process runner and
hardware.

Copy counters report logical operation volume, not hardware DRAM traffic.
Dense volume includes old K/V copied during each request-cache growth and live
K/V copied into a newly padded batch tensor at every decode step. Paged volume
includes old physical pages copied only when a layer's geometric pool grows.
Appending a token writes only its page slice, and later requests reuse freed
pages.

The direct-paged median reaches 1,116 live pages out of 1,152 reserved pages,
reuses 2,196 page allocations, and records 15,840 unused tail slots across
layer caches. It grows the layer pools 144 times because the measured run
starts empty. These counters make reuse and fragmentation visible; static
single-request latency cannot.

The workload validates continuous batching, chunked prefill, incremental
growth, and page reuse. Prefix sharing and speculative decoding require
separate traces with shared prefixes or cache rewind events and are not claimed
by this result.

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
