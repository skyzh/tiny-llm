# 🚧 Appendix: Performance by Chapter

> 🚧 This appendix is under review and may change.

This appendix connects each chapter to a measurable performance or serving
effect. It distinguishes **latency and throughput** from **usability**. Paged
allocation, chunked prefill, and continuous batching can make a server more
usable under load even when a teaching kernel is slower for one request.

The percentages below are not additive. Replacing one bottleneck changes the
fraction of time spent in every other operator. Use synchronized end-to-end
measurements to judge a checkpoint, and use one-factor ablations only to
explain why it moved.

## Reproduce the Checkpoint Comparison

Run all checkpoints in fresh, sequential processes and report the median:

```bash
pdm run bench-course-progression --offline --repeats 3 \
  --model qwen3-0.6b --input-len 128 --output-len 65 --warmup 2
```

For the chapter-by-chapter Week 2 ladder, run:

```bash
pdm run bench-week2-progression --offline --repeats 3 \
  --model qwen3-0.6b --input-len 128 --output-len 65 --warmup 2
```

The runner prints prefill and decode throughput, speed relative to Week 1, and
the remaining gap to MLX. It alternates checkpoint order across repeats to
reduce systematic thermal bias. It cannot isolate the machine for you: stop
other CPU/GPU workloads, keep power mode fixed, and use `--cooldown-seconds`
when the system does not return to a stable temperature between processes.
Add `--json-output course-performance.json` to retain every sample, the
medians, seed, selected variants, MLX version, and non-identifying hardware
details.

The prefill timer computes logits for every prompt position on Week 1, Week 2,
Week 3, and MLX. Decode requests only the final row, but its input already has
one token, so that shortcut does not change the compared decode work.

The default uses the reference solution. Add `--solution tiny_llm` to run the
same checkpoint sequence against your implementation.

For operator attribution and serving throughput, also use:

```bash
pdm run bench-week2-operators --model qwen3-0.6b --context 128

pdm run bench --solution tiny_llm_ref --loader week3 --batch-decode \
  --num-seqs 16 --batch-size 4 --model qwen3-0.6b
```

Always record the model, MLX version, hardware, prompt/output lengths, warmup,
repeat count, and whether the number is a median or one representative run.

## Course Optimization Map

Use this table to navigate from each optimization concept to its implementation
boundary and teaching chapter. Complete the rows in chapter order; use the Week
3 serving lab to record scheduler and cache experiments that you decide not to
keep.

| Course step | Required implementation boundary | Where it is explained |
|---|---|---|
| Synchronized, matched measurement | `bench.py` evaluates timed work, computes full prompt logits for matched prefill, releases caches, and separates prefill from decode. `bench_course_progression.py` uses fresh processes, alternating order, medians, and optional pauses between runs. | Week 2 Day 2 and this appendix |
| KV cache before decode tuning | `TinyKvFullCache`, `create_kv_cache`, and cached generation prefill once and then pass one new token. | Week 2 Day 1 |
| Packed 4-bit weights | `QuantizedWeights` preserves weight, scale, bias, group size, and bit width instead of materializing dense 16-bit matrices. | Week 2 Day 3 |
| Shape-dispatched SIMD matvec | `quantized_matvec_x2` handles ordinary projections and `quantized_matvec_x8` handles `K >= 8192`; eight SIMD groups share activation loads and reduce with `simd_sum`. | Week 2 Day 3 |
| Matvec scheduling cleanup | Reread the cache-hot activation vector instead of copying it to threadgroup memory and waiting at a barrier. | Week 2 Day 3 scheduling experiments |
| Online decode attention | `decode_attention_custom` keeps the query in registers, uses 32 SIMD groups, FP32 online softmax, and a parallel final reduction for `L <= 8` and context at most 256. | Week 2 Day 4 |
| Fused RMSNorm | One 256-thread group performs a two-level reduction, normalization, and learned scaling with FP32 accumulation. | Week 2 Day 5 |
| Fused RoPE | One thread computes an angle once for a pair and reuses it across four heads; batch offsets are normalized once per model call. | Week 2 Day 5 |
| Fused SwiGLU | One Metal dispatch computes SiLU and multiplies the up branch without graph intermediates. | Week 2 Day 5 |
| SIMD-matrix quantized prefill | `quantized_matmul_simdgroup` uses 8-by-8 matrix fragments, unpacking weights directly and accumulating in FP32. | Week 2 Day 6 |
| Direct quantized embedding | The embedding kernel fuses row gather, int4 unpacking, scale, and bias for both signed and unsigned token IDs. | Week 2 Day 6 |
| Decode graph cleanup | Generation requests `logits_to_keep=1`, and single-token decode omits the causal-mask graph. | Week 2 Day 6 |
| Paged FlashAttention | The BF16 SIMD-matrix kernel streams K/V from page storage, uses online softmax, bounds causal work, and avoids both a dense gather and the quadratic score tensor. | Week 3 Day 5 |
| Serving and cache scheduling | Continuous batching reuses active slots, chunked prefill bounds scheduler stalls, and paged attention separates logical context from reusable physical pages. | Week 3 Days 1–4 |

The model-loading boundary is not an optimization shortcut. MLX still supplies
model files, arrays, streams, allocation, and extension dispatch; the Week 2
learned operators above execute course-owned Python, C++, or Metal code.

## Reference End-to-End Checkpoints

Use the following Qwen3-0.6B snapshot as a reasonableness check after running
the commands above. It was measured on an Apple M4 Pro with a 20-core GPU and
64 GB of memory, using a 128-token prompt, 65 generated tokens, two complete
warmups, and a three-process median. Some rows come from separate historical
campaigns and are labeled accordingly; rerun the current progression command
for a matched comparison on your machine.

| Checkpoint | Prefill tok/s | Versus Week 1 | Gap to MLX | Decode tok/s | Versus Week 1 | Gap to MLX |
|---|---:|---:|---:|---:|---:|---:|
| Week 1 readable model | 3,342.45 | baseline | 24.1% slower | 19.49 | baseline | 93.4% slower |
| Week 2 through Day 5 | 1,012.27 | 69.7% slower | 77.0% slower | 240.17 | 12.32x | 18.5% slower |
| Week 2 completed SIMD-matrix model (historical dense run) | 2,042 | 38.9% slower | 53.6% slower | 246 | 12.62x | 16.5% slower |
| Week 3 completed paged FlashAttention model | 1,956.79 | 41.5% slower | 55.6% slower | 204.80 | 10.51x | 30.5% slower |
| MLX | 4,402.94 | 1.32x | baseline | 294.75 | 15.12x | baseline |

Week 1 prefill is fast because it materializes dense 16-bit weights and uses
efficient dense matrix multiplication. Week 2 deliberately optimizes the
memory-bound one-token decode shape first; its readable scalar quantized
prefill path therefore regresses. Week 2 Day 6 recovers part of that prefill
gap before serving work begins. The paged vector decode kernel stays close to the
dense course kernel, while page-cache and serving-runtime overhead leave the
single-request checkpoint about 13.8% behind Week 2 decode. The runner releases
every request cache, including warmups, so paged checkpoints reuse allocator
capacity rather than timing pool growth left by an earlier run.

## Historical Week 2 Chapter Ladder

The next two tables preserve a separate three-process M4 Pro Week 2 progression
run with a 128-token prompt, 65 generated tokens, and two warmups. Its Week 1
decode baseline was 19.51 tok/s and its matched MLX denominator was 327.84
tok/s. Use this campaign to attribute the cumulative Week 2 steps; do not mix
its denominators with the newer full-course snapshot above.

### Week 1: Establish the Readable Baseline

| Chapter | Performance or usability effect | Comparison |
|---|---|---|
| 1.1 Attention | Materializes scores and probabilities so the algorithm is inspectable. | No speed claim; this is the readable attention baseline. |
| 1.2 RoPE | Builds positions, sine, and cosine with array operations. | No isolated checkpoint gain; Week 2 later fuses this work. |
| 1.3 GQA | Shares K/V heads, reducing KV storage versus full multi-head attention. | The Qwen3 architecture already fixes the head ratio, so the course has no MHA-to-GQA timing ablation. |
| 1.4 RMSNorm and MLP | Uses readable array expressions and exposes intermediate tensors. | No speed claim; Week 2 measures fused replacements against these equations. |
| 1.5 Qwen3 Model | Produces the first complete model and the baseline prefill graph. | The completed Week 1 checkpoint measured about 3,350 prefill tok/s, 24.1% below MLX. |
| 1.6 Generation | Recomputes the growing prefix for every output token. | About 19.51 decode tok/s at context 128, roughly 94.0% below MLX. |
| 1.7 Sampling | Adds temperature/top-p/top-k policy after logits. | Not a model-throughput optimization; deterministic argmax is used by the course benchmark. |

### Week 2: Build the Fast Decode Checkpoint

Unless a row identifies an isolated or M1 Pro measurement, the estimates come
from this historical Week 2 progression campaign. End-to-end rows are
cumulative and include all earlier rows; do not add the percentages together.

| Chapter | Change | Measured estimate | How to interpret it |
|---|---|---|---|
| 2.1 KV Cache | Prefill once, retain K/V, and process one new query per decode step while preserving the readable Week 1 operators. | 19.51 → 101.80 decode tok/s: 5.22x Week 1 and 68.9% below MLX. | This is an algorithmic gain, not a faster operator. Its size depends strongly on context and output length and grows as full-prefix recomputation becomes more expensive. |
| 2.2 Benchmark | Synchronize lazy work, separate prefill from decode, and establish the cached and MLX baselines. | 0% direct speedup. | Measurement prevents changes in inputs, synchronization, or machine state from being mistaken for optimization. |
| 2.3 Quantized Matvec | Keep weights packed and use the course SIMD matvec for every decode projection. | End to end: 101.80 → 134.24 tok/s (+31.9%), 6.88x Week 1 and 59.1% below MLX. Isolated versus the scalar quantized kernel: Q/K/V/O were 1.39–1.65x, gate/up/down were 1.86–1.89x, and the 151,936-row tied output head was about 10.5x. | Operator and end-to-end gains answer different questions. On the earlier M1 Pro path, the first two-column SIMD matvec added about 5.7%, and removing its activation barrier added another 3–4%; those historical deltas are not part of the M4 Pro ladder. |
| 2.4 Decode Attention | Replace the readable float32 score/softmax/value composition with a course-owned online-softmax kernel in its measured short-context range. | 134.24 → 143.42 tok/s (+6.8%) at context 128: 7.35x Week 1 and 56.3% below MLX. | An isolated comparison against a different readable bfloat16 composition was about 7% faster at context 128 but became slower at longer contexts; it is not the cumulative course comparison. Quantized weight reads still dominate short decode, and the fixed 32-group schedule does not scale monotonically. Report dtype, context, schedule, and end-to-end throughput rather than treating avoided intermediates as a speedup by themselves. |
| 2.5a RMSNorm | Fuse square, reduction, normalization, and scaling in a course Metal kernel. | Isolated: about 1.36x the readable equation. Cumulative: 143.42 → 193.18 tok/s (+34.7%). | The cumulative delta includes every preceding chapter, while the isolated number compares only the operator at decode shapes. |
| 2.5b RoPE | Rotate pairs directly and reuse each angle across four heads. | Cumulative: 193.18 → 222.51 tok/s (+15.2%). | This row starts from the RMSNorm checkpoint, so it is not additive with the RMSNorm percentage. The prior isolated RoPE number used a mismatched MLX layout and is intentionally omitted until rerun with the corrected transpose. |
| 2.5c SwiGLU | Fuse SiLU and multiplication into one element-by-element kernel. | Isolated: about 1.39x the readable equation. Cumulative: 222.51 → 242.67 tok/s (+9.1%). | This row starts from the RoPE checkpoint. The verified isolated RMSNorm and SwiGLU gains were about 36% and 39% over their readable equations. |
| 2.6 SIMD-Matrix Prefill | Add the tiled quantized prefill schedule, direct quantized embedding, and last-token projection. | A separate dense ablation reached about 2,042 prefill tok/s and 246 decode tok/s. | This is not denominator-compatible with the preceding progression rows; rerun the current ladder for a matched result. |
| Week 2 checkpoint | Combine the dense KV cache, SIMD matvec, decode attention, fused elementwise kernels, and SIMD-matrix prefill. | The through-Day-5 campaign reached 242.67 decode tok/s versus 327.84 for matched MLX; the later Day 6 dense ablation reached about 246 tok/s. | Compare rows only within one campaign; use the current commands above for the latest integrated checkpoint. |

## Week 3: Improve Serving Structure

| Chapter | What improves | Performance reality |
|---|---|---|
| 3.1 Continuous Batching | Reuses decode slots as requests finish and keeps multiple requests active. | Improves aggregate utilization, not single-request latency. In one four-request snapshot, dense batching reached 305.3 aggregate decode tok/s versus 240.2 for one Week 2 request; do not treat different batch sizes as a kernel speedup. |
| 3.2 Chunked Prefill | Bounds how long a prompt can delay active decoders. | Primarily improves fairness and time-to-next-token. Smaller chunks add launches and may reduce aggregate throughput; report both latency and throughput. |
| 3.3 Paged Attention, Part 1 | Allocates a paged KV cache with fixed-size reusable pages and separates logical context from physical storage. | Per-layer pools grow geometrically and share storage across requests in that layer. This improves capacity, reuse, removal, and fragmentation without serializing every layer through one backing tensor. |
| 3.4 Paged Attention, Part 2 | Reads K/V through page tables without rebuilding dense per-request K/V. | The decode operator is approximately matched with the dense course kernel across the documented batch/context cases. The single-request paged checkpoint reached 206.96 decode tok/s, 29.8% below MLX; the matched four-request serving run reached 348.75 aggregate decode tok/s. |
| 3.5 Paged FlashAttention | Tiles the direct Day 4 page walk with Week 2 SIMD-matrix fragments and online softmax. | Its opportunity is long prefill; decode retains the Day 4 vector schedule. The completed historical snapshot reached about 1,957 prefill tok/s at the short course shape. |
| 3.6 Optional MoE | Routes tokens through selected experts and supports sparse Qwen3 variants. | Expands model coverage and can reduce active parameter work, but the course has no controlled dense-versus-MoE speed claim. |
| 3.7 Optional Serving Lab | Varies chunk size, batch size, page size, request mix, and dense-versus-paged policy without adding a model kernel. | Report latency, throughput, memory, fairness, and allocator behavior together. |
| Optional Speculative Decoding | Drafts several tokens and verifies them with the target model. | Work in progress. Speed depends on acceptance rate and cache-rewind cost; no result should be claimed yet. |

### Moved Week 2 Kernel Attribution

These measurements motivated the kernels that now live in Week 2 Day 6. They
are reference evidence, not results to copy. Reproduce the comparison with the
progression and operator commands at the start of this appendix.

Last-token logits made the output projection about 40x faster and the complete
Qwen3-0.6B model about 1.29x faster in one M4 Pro measurement. Normalizing RoPE
offsets once saved about 2% per isolated call, while omitting the single-token
causal flag was below the measurable noise floor.

An earlier dense Week 2-based ablation reached about 2,042 prefill tok/s and
246 decode tok/s, compared with about 3,318 prefill tok/s and 19.4 decode tok/s
for Week 1. This isolates the lab kernels from paging and is not the result of
the integrated Week 3 loader.

On the completed paged stack after hoisting quantization parameters, a matched
three-process run reached about 2,371 prefill tok/s and 210 decode tok/s, versus
4,416 prefill tok/s and 329 decode tok/s for MLX. The lab changes prefill matrix
products and should not materially move decode.

The following cached-decode rows are reverse ablations from the finished model.
Each comparison loads the same weights, alternates optimized and readable runs,
and changes only the named component:

| Replacement | Readable tok/s | Optimized tok/s | Throughput gain |
|---|---:|---:|---:|
| Quantized embedding gather | 243.87 | 245.75 | +0.8% |
| RMSNorm Metal kernel | 185.28 | 246.16 | +32.9% |
| RoPE Metal kernel | 193.38 | 245.92 | +27.2% |
| Fused SwiGLU | 219.14 | 245.29 | +11.9% |
| Online decode attention | 245.17 | 245.73 | +0.2% |

These gains are not additive. As a diagnostic only, replacing the course QMV
with MLX improved throughput by 14.5%, replacing attention improved it by 9.8%,
and replacing both reached about 309.7 tok/s. Those MLX substitutions are not
part of the course solution; they identify the two course-owned kernels with
the largest remaining optimization ceiling.

In one matched four-request snapshot, the dense path reached 305.31 aggregate
decode tok/s while the Week 3 paged path reached 348.75 tok/s. This is not a
universal paged-kernel speedup: it is an end-to-end serving result that includes
avoided repacking and cache reuse. The isolated operator remains close to the
dense course kernel and behind MLX.

## Week 4: Measure Application Quality Separately

| Chapter | Appropriate metric |
|---|---|
| 4.1 Coding Agent | Task-completion rate, steps, generated tokens, tool errors, and wall-clock latency. The agent loop adds work; it does not improve model tok/s. |
| 4.2 RAG | Retrieval recall/precision, grounded-answer quality, added prompt tokens, and time to first token. Better evidence can improve answers while reducing throughput. |
| 4.3 Tool Calling and Agent Serving | Valid-action rate, recovery rate, concurrent sessions, cache reuse, and end-to-end task latency. Keep these separate from kernel throughput. |

The Week 4 chapters remain works in progress. Their success criterion is useful,
safe application behavior on top of the Week 2/3 inference interfaces, not a
higher isolated token rate.

{{#include copyright.md}}
