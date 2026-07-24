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

`--prefill-logits last` is a generation-serving workload: both the reference
solution and MLX project only the last prompt row into vocabulary logits. Use
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

The measured machine below is an Apple M4 Pro with a 20-core GPU and 64 GB of
memory. Static Week 2 rows use two complete warmups; the continuous-serving
rows use one. Both report the median of three fresh alternating processes.

## Long-Context Budget for Week 4

Context length has separate model, memory, and latency limits. For the course
Qwen3-4B checkpoint, one token of BF16 K/V state occupies

```text
36 layers * 2 (K and V) * 8 KV heads * 128 values * 2 bytes
    = 147,456 bytes = 144 KiB per token
```

The checkpoint declares `max_position_embeddings = 65,536`, but its
`rope_scaling` field is empty. Qwen documents that Qwen3 pretraining covers
[32,768 tokens](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md#context-length)
and recommends RoPE scaling for substantially longer inputs. The unmodified
course model therefore has a 32,768-token validated limit even though its
configuration permits a larger position experiment.

Memory is not the binding limit on the measured 64 GB M4 Pro. MLX reports a
51.84 GiB recommended GPU working set, and the quantized checkpoint occupies
1.99 GiB. Reserving 8 GiB for activations, allocator slack, and outputs gives

```text
floor((51.84 GiB - 1.99 GiB - 8 GiB) / 144 KiB) = 304,738 tokens
```

That estimate is a capacity calculation, not permission to exceed the model's
trained range. The course limit is the minimum of the limits:

```text
min(32,768 trained, 65,536 configured, 304,738 memory) = 32,768 tokens
```

Week 4 uses 32,768 total tokens as its hard context budget. It starts
compaction before the rendered input exceeds 24,576 tokens, reserving 8,192
tokens for the next model response and a large tool result. The tokenizer must
count the complete rendered request, including system instructions and tool
schemas.

### What Becomes Slow at 300K

FlashAttention removes the quadratic score-matrix allocation; it does not
remove the work. Full-attention prefill remains quadratic in context length,
so 300K contains about 84 times the attention work of 32K. One-token decode
must read a linearly growing K/V history at every layer.

The following synthetic operator sweep uses MLX 0.32.0, one Qwen3-4B-shaped
BF16 decode query, three fresh processes, and the median of fifteen synchronized
dispatches per process. The final column sums the isolated layer latency across
36 layers and is an optimistic attention-only ceiling; a complete model must
also run projections, normalization, sampling, and cache updates.

| Context | Full-model BF16 KV | MLX SDPA per layer | Attention-only decode ceiling |
|---:|---:|---:|---:|
| 2,048 | 0.28 GiB | 0.14 ms | 195.33 tok/s |
| 8,192 | 1.12 GiB | 0.29 ms | 96.72 tok/s |
| 32,768 | 4.50 GiB | 0.92 ms | 30.28 tok/s |
| 65,536 | 9.00 GiB | 1.73 ms | 16.08 tok/s |
| 131,072 | 18.00 GiB | 3.65 ms | 7.61 tok/s |
| 300,000 | 41.20 GiB | 9.49 ms | 2.93 tok/s |

The 300K operator allocation runs on this M4 Pro, but an end-to-end 300K run of
the course checkpoint would be outside its configured and pretraining ranges,
would leave little working-set headroom, and would make initial prefill
impractical. It is useful as a kernel stress test, not as a supported course
context.

MLX contains several long-context optimizations. Its fused GQA decode path
automatically switches to a context-partitioned two-pass reduction; the
[0.30.4 release](https://github.com/ml-explore/mlx/releases/tag/v0.30.4)
specifically calls out faster long-context vector GQA. Multi-token attention
uses a tiled fused path, and MLX-LM chunks prompt evaluation to bound temporary
activations. MLX-LM also offers prompt-prefix reuse, a rotating fixed-size
cache, and quantized KV storage. Prefix reuse helps repeated prompts; cache
rotation changes full-attention semantics; and KV quantization trades numerical
precision and sometimes speed for capacity. None makes the first full 300K
prefill linear-time.

Reproduce the operator sweep with:

```bash
pdm run bench-long-context-attention \
  --json-output benchmark_results/m4-pro-qwen3-4b-long-context-mlx-0.32.0.json
```

## Dependency Upgrade

The project upgraded from MLX 0.29.1 to 0.32.0 and from the mlx-lm 0.28 series
to 0.31.3. A matched Qwen3-4B run showed:

| Context | Metric | MLX 0.29.1 | MLX 0.32.0 | Change |
|---:|---|---:|---:|---:|
| 128 | Prefill tok/s | 825.48 | 828.34 | +0.35% |
| 128 | Decode tok/s | 88.32 | 88.08 | -0.27% |
| 2,048 | Prefill tok/s | 816.73 | 820.85 | +0.50% |
| 2,048 | Decode tok/s | 78.42 | 74.81 | -4.60% |

The small differences show why the comparison must record exact dependency
versions: the MLX denominator is part of the experiment, even when an upgrade
does not materially change the result.

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
| Day 4 | Fused decode attention | 104.51 | 60.23 | 38.50 | Replace growing score/softmax/value work. |
| Day 5 | Fused model kernels | 106.03 | 77.09 | 44.98 | Remove the newly exposed repeated graph launches. |
| Day 6 | SIMD-matrix prefill | 793.15 | 76.98 | 70.69 | Fix the quantized matrix path exposed by Day 3. |
| Day 7 | Split-K prefill | 792.18 | 77.41 | 71.05 | Fill the GPU only for under-occupied short projections. |
| Baseline | MLX 0.32.0 | 827.74 | 87.58 | 79.81 | External denominator. |

### The Kernel Profile That Selects Each Chapter

The reference-solution profile does not replace an operator with an MLX
operator. It calls the projection, attention, pointwise, and cache paths from
`tiny_llm_ref` at Qwen3-4B shapes and replays each group at the model's real
dispatch count. The projection replay preserves the transformer dependency
order so work from a later MLP cannot hide an under-filled attention
projection. Each category uses
one synchronization, and the median follows five warmups and fifteen samples:

```bash
pdm run profile-week2-kernels --model qwen3-4b \
  --warmup 5 --iterations 15 \
  --json-output week2-kernel-profile.json
```

The bar widths below are normalized within a checkpoint. The time at the right
is the sum of the synchronized category medians, not a throughput measurement.
Forcing category boundaries prevents some whole-graph fusion, so use the shares
to rank work and the fresh-process checkpoint table above to accept or reject a
change. The attributed totals follow the complete-model direction and range
from 1.01× to 1.28× its phase time.

![Flat flame chart of Week 2 kernel-time attribution](./week2-kernel-profile.svg)

The profile makes the progression concrete:

- Cached decode spends 80.3% of attributed time in dense projections. Day 3
  therefore changes weight storage and the decode projection schedule first.
- After packed matvec, the pointwise group is 39.3% while attention is only
  5.2% at the 128-token acceptance context. Day 4 is a deliberately scoped
  online-softmax and context-scaling lesson, not the main short-context gain.
  Its follow-up profile leaves the 37.4% pointwise group for Day 5.
- After Day 5, decode clears the course target. Changing the workload to
  128-token prefill makes the readable quantized projection path 99.0% of
  attributed time, which selects the cooperative matrix kernel in Day 6.
- After Day 6, projections remain most of the inherent prefill work, but the
  long-shape operator comparison is already close to MLX. The 32-token shape
  sweep then isolates under-occupied Qwen projections and selects Split-K only
  below their measured crossover.

The checked-in raw profile is
`benchmark_results/m4-pro-qwen3-4b-week2-kernel-profile-mlx-0.32.0.json`.

### Day 1: Cache the Prefix

The dense cache makes prefill a one-time cost, but every decode projection
still reads dense weights. Day 1 therefore starts with respectable prefill and
only 24.51 decode tok/s. The result gives Day 2 a real cached baseline to
profile.

### Day 2: Measure Before Optimizing

Day 2 changes the measurement discipline rather than the model. Synchronized
kernel-group timings attribute 80.3% of the cached decode profile to dense
projections. That evidence selects packed quantized matvec for Day 3; the CLI
trace workflow remains useful when a compatible Shader Timeline is available.

### Day 3: Keep Weights Packed

The x4 W4A16 matvec raises decode from 24.51 to 59.17 tok/s, a 141.4% gain.
Prefill intentionally falls from 726.25 to 104.82 tok/s because the packed
checkpoint still sends matrix-shaped inputs through the readable one-thread
quantized kernel. This is not hidden as a temporary implementation detail: the
new prefill profile makes quantized matrix multiplication the next dominant
cost.

### Day 4: Fused Decode Attention

At the fixed short context, attention is 5.2% of the attributed Day 3 profile.
The online-softmax kernel reduces its measured group time from 1.12 to 0.94 ms,
and complete-model decode rises by only 1.8%. Context sweeps determine its
retained dispatch range. The small acceptance gain prevents the course from
incorrectly presenting attention as the main short-context bottleneck.

### Day 5: Fused Model Kernels

Day 5 applies three independently measurable changes. Fast RMSNorm raises
decode by 11.1%, fast RoPE adds 8.2%, and fused SwiGLU adds another 6.5% at
their cumulative checkpoints. The pointwise group falls from 37.4% after Day 4
to 10.7%, and the completed day reaches 77.09 decode tok/s. The course target
is now met for decode, so the next profile switches to prefill.

### Day 6: Use Cooperative Loads for Quantized Prefill

At the Day 5 prefill checkpoint, projections in the reference solution account
for 1,259.90 ms of the 1,272.06 ms attributed profile, or 99.0%. Attention
accounts for 5.94 ms, and normalization, position, and activation together
account for 6.21 ms.
This direct profile selects quantized matrix multiplication without routing any
model operation through MLX's quantized-matmul implementation.

Assign contiguous activation elements to adjacent lanes and stage them with
aligned cooperative block loads. Combined with a 32×32×32 tile, this schedule
makes the large Qwen3-4B projections essentially match MLX:

| Projection at `M=2048` | Reference solution | MLX |
|---|---:|---:|
| Q, `2560 -> 4096` | 6.66 ms | 6.72 ms |
| K, `2560 -> 1024` | 1.84 ms | 1.85 ms |
| MLP gate, `2560 -> 9728` | 15.49 ms | 15.65 ms |
| MLP down, `9728 -> 2560` | 15.66 ms | 15.76 ms |

The cooperative schedule raises prefill from 106.03 to 793.15 tok/s, a 648.0%
gain, while leaving vector decode unchanged.

### Day 7: Split K Only Below the Crossover

The Day 6 shape sweep finds under-filled result grids at short `M`. Day 7
splits K until the result grid approaches 320 threadgroups, using at most 16
equal, 128-aligned partitions. The narrow K/V projections receive the largest
split; Q, O, and MLP down may also split at `M=32`, while the wide gate/up grid
is already occupied. At the 32-token control point:

| Checkpoint | Prefill tok/s | Decode tok/s | Prefill / MLX |
|---|---:|---:|---:|
| Day 6 cooperative matmul | 586.12 | 78.03 | 83.3% |
| Day 7 split-K | 650.69 | 77.89 | 92.5% |
| MLX 0.32.0 | 703.58 | 84.69 | 100% |

Split-K adds 11.0% prefill at this short shape and no decode gain. At the
128-token acceptance shape only the narrow K/V projections retain a two-way
split; the other major projections fall back to Day 6, and the complete-model
result remains neutral. At 2,048 tokens every projection falls back.

The fresh-process samples for this control point are checked in at
`benchmark_results/m4-pro-qwen3-4b-week2-32-mlx-0.32.0.json`.

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
| Day 2 | Chunked admission with dense reconstruction | 653.24 prefill; 32.77 output; 53.99 decode tok/s | Establishes the dense serving baseline. |
| Day 3 | Paged storage with compatibility gather | 662.69 prefill; 38.38 output; 71.02 decode tok/s | +17.1% output; +31.5% decode; -50.6% copy volume. |
| Day 4 | Direct paged decode schedule | 100.35 aggregate decode tok/s | +41.3% decode over the compatibility gather path. |
| Day 5 | Complete direct paged path | 650.10 prefill; 45.05 output; 100.35 decode tok/s | +37.4% output and request throughput over dense serving. |

Day 1 introduces scheduling, not a kernel speedup. Day 2 makes the hidden cost
measurable: appending one token still reconstructs a padded dense batch. Day 3
makes pages canonical but retains `gather_dense()` as a compatibility
checkpoint. Days 4 and 5 then remove that compatibility movement for decode
and long-query prefill respectively.

Days 4 and 5 share the final direct-paged process: queries with `L <= 8`
dispatch to the Day 4 decode schedule, while longer chunks dispatch to the Day
5 tiled schedule. The phase timers report their decode and prefill throughput
inside the same request trace; they are not results from different workloads.

Every headline number above comes from the same continuous-batch campaign. The
cumulative serving endpoints are:

| Storage and attention path | Prefill tok/s | Output tok/s | Decode tok/s | Requests/s | Peak KV MiB | Avoidable KV copy MiB |
|---|---:|---:|---:|---:|---:|---:|
| Dense growth and reconstruction | 653.24 | 32.77 | 53.99 | 0.437 | 1,096 | 209,532 |
| Paged storage plus dense gather | 662.69 | 38.38 | 71.02 | 0.511 | — | 103,445 |
| Direct paged attention | 650.10 | 45.05 | 100.35 | 0.600 | 576 | 504 |

The compatibility row omits peak storage because an exact peak must include
both the page pool and temporary dense staging allocation. Its other counters
remain directly comparable.

Direct paged attention improves output and request throughput by 37.4%,
aggregate decode by 85.9%, and peak KV storage by 47.4% relative to dense
serving. Avoidable logical copy volume falls by 99.8%. Relative to paged
storage plus gather, the direct operator adds 17.4% output throughput, 41.3%
decode throughput, and removes 99.5% of the remaining copy volume. Prefill is
0.5% below dense and 1.9% below gather at the 128-token serving chunk, so the
chapter does not claim a short-chunk FlashAttention speedup.

The 8K static run remains a secondary kernel diagnostic, not a Week 3 headline
or acceptance result. At that shape, paged FlashAttention raises prefill from
384.88 to 427.01 tok/s. MLX reaches 568.74 tok/s, so the page-aware path in the
reference solution reaches 75.1%. This explains where query tiling begins to
help without mixing a static denominator into the serving progression.
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

## GPU Profile and CLI Tools

The synchronized reference-solution kernel attribution used above is available
without a GUI:

```bash
pdm run profile-week2-kernels --model qwen3-4b \
  --warmup 5 --iterations 15
```

It is a flat operator profile, not a CPU call-stack flame graph. When Shader
Timeline export works on the target hardware, validate the same ranking with a
full-model Metal trace.

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

The stock Metal System Trace exposes queues and command buffers on the measured
M4 Pro, but its Shader Timeline is disabled and the selected Metal
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
