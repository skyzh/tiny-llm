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

Generation serving needs only the final prompt logit row. Measure that workload
separately (Week 1 is excluded because its readable interface always computes
all rows):

```bash
pdm run bench-course-progression --offline --repeats 3 \
  --variant week2 --variant week3 --variant mlx \
  --prefill-logits last --input-len 512 --output-len 65 --warmup 2
```

Do not compare a final-row tiny-llm run with an all-row MLX run. The progression
runner passes the same prefill mode to every selected checkpoint and records it
in JSON output.

The default uses the reference solution. Add `--solution tiny_llm` to run the
same checkpoint sequence against your implementation.

For operator attribution and serving throughput, also use:

```bash
pdm run bench-week2-operators --model qwen3-0.6b --context 128

pdm run bench --solution tiny_llm_ref --loader week3 --batch-decode \
  --num-seqs 16 --batch-size 4 --prefill-step 128 \
  --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 1 \
  --model qwen3-0.6b
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
64 GB of memory, MLX 0.29.1, and Python 3.12.13. Every row uses a 128-token
prompt, 65 generated tokens, two complete warmups, and the median of three
fresh processes. The runner alternated checkpoint order across repeats.

| Checkpoint | Prefill tok/s | Versus Week 1 | Gap to MLX | Decode tok/s | Versus Week 1 | Gap to MLX |
|---|---:|---:|---:|---:|---:|---:|
| Week 1 readable model | 3,310.10 | baseline | 24.5% slower | 19.44 | baseline | 93.9% slower |
| Week 2 completed SIMD-matrix model | 1,928.49 | 41.7% slower | 56.0% slower | 244.87 | 12.60x | 23.5% slower |
| Week 3 completed paged FlashAttention model | 1,859.97 | 43.8% slower | 57.6% slower | 207.40 | 10.67x | 35.2% slower |
| MLX | 4,383.47 | 1.32x | baseline | 320.06 | 16.46x | baseline |

Week 1 prefill is fast because it materializes dense 16-bit weights and uses
efficient dense matrix multiplication. Week 2 deliberately optimizes the
memory-bound one-token decode shape first; its scalar quantized prefill path
therefore regresses until Week 2 Day 6 introduces SIMD-matrix fragments. Week
3's page-table and serving-runtime work is not free: the single-request
checkpoint is 15.3% slower than Week 2 for decode and 3.6% slower for prefill at
this short shape. The serving measurement below shows the separate benefit
when four requests share the decode batch.

## Matched Week 2 Chapter Ladder

This separate three-process campaign uses the same model, prompt/output shape,
warmups, hardware, and alternating order. Its matched denominators are Week 1
at 3,313.05 prefill and 19.34 decode tok/s, and MLX at 4,373.56 prefill and
316.35 decode tok/s. End-to-end rows are cumulative; percentages are changes
from the preceding executable checkpoint unless stated otherwise.

### Week 1: Establish the Readable Baseline

| Chapter | Performance or usability effect | Comparison |
|---|---|---|
| 1.1 Attention | Materializes scores and probabilities so the algorithm is inspectable. | No speed claim; this is the readable attention baseline. |
| 1.2 RoPE | Builds positions, sine, and cosine with array operations. | No isolated checkpoint gain; Week 2 later fuses this work. |
| 1.3 GQA | Shares K/V heads, reducing KV storage versus full multi-head attention. | The Qwen3 architecture already fixes the head ratio, so the course has no MHA-to-GQA timing ablation. |
| 1.4 RMSNorm and MLP | Uses readable array expressions and exposes intermediate tensors. | No speed claim; Week 2 measures fused replacements against these equations. |
| 1.5 Qwen3 Model | Produces the first complete model and the baseline prefill graph. | The completed Week 1 checkpoint measured 3,313.05 prefill tok/s, 24.2% below MLX. |
| 1.6 Generation | Recomputes the growing prefix for every output token. | The matched checkpoint reached 19.34 decode tok/s at context 128, 93.9% below MLX. |
| 1.7 Sampling | Adds temperature/top-p/top-k policy after logits. | Not a model-throughput optimization; deterministic argmax is used by the course benchmark. |

### Week 2: Build the Fast Decode Checkpoint

| Chapter | Change | Measured estimate | How to interpret it |
|---|---|---|---|
| 2.1 KV Cache | Prefill once, retain K/V, and process one new query per decode step while preserving the readable Week 1 operators. | Prefill 3,296.76; decode 98.47 tok/s, or 5.09x Week 1 and 68.9% below MLX. | This is an algorithmic gain, not a faster operator. Its size depends strongly on context and output length and grows as full-prefix recomputation becomes more expensive. |
| 2.2 Benchmark | Synchronize lazy work, separate prefill from decode, and establish the cached and MLX baselines. | 0% direct speedup. | Measurement prevents changes in inputs, synchronization, or machine state from being mistaken for optimization. |
| 2.3 Quantized Matvec | Keep weights packed and use the course SIMD matvec for every decode projection. | Prefill 747.08; decode 130.19 tok/s, +32.2% from Day 1 and 58.8% below MLX. | Packed weights reduce decode memory traffic but make the still-scalar prefill path much slower. The isolated table below separates kernel latency from this end-to-end result. |
| 2.4 Decode Attention | Replace the readable score/softmax/value composition with a course-owned online-softmax kernel in its measured short-context range. | Prefill 732.02; decode 137.71 tok/s, +5.8%. | Quantized weight reads still dominate short decode. Report dtype, context, schedule, and end-to-end throughput rather than treating avoided intermediates as a speedup by themselves. |
| 2.5a RMSNorm | Fuse square, reduction, normalization, and scaling in a course Metal kernel. | Prefill 731.85; decode 185.55 tok/s, +34.7%. | The cumulative delta includes all prior chapters; the isolated kernel is 1.31x faster than the readable equation. |
| 2.5b RoPE | Rotate pairs directly and reuse each angle across four heads. | Prefill 766.62; decode 214.61 tok/s, +15.7%. | The corrected-layout isolated comparison is 1.52x faster than readable RoPE. |
| 2.5c SwiGLU | Fuse SiLU and multiplication into one element-by-element kernel. | Prefill 777.80; decode 234.78 tok/s, +9.4%. | The isolated fused kernel is 1.38x faster than the readable expression. |
| 2.6 SIMD-Matrix Prefill | Add the tiled quantized prefill schedule, direct quantized embedding, and last-token projection. | Prefill 1,932.94 tok/s, +148.5%; decode 240.98 tok/s, +2.6%. | This is the workload-shape handoff: matrix fragments repair prefill while leaving the one-token SIMD matvec path intact. |
| Week 2 checkpoint | Combine the dense KV cache, SIMD matvec, decode attention, fused element-wise kernels, and SIMD-matrix prefill. | 12.46x Week 1 decode; 23.8% below MLX decode and 55.8% below MLX prefill. | All figures now come from one matched campaign. |

## Week 3: Improve Serving Structure

| Chapter | What improves | Performance reality |
|---|---|---|
| 3.1 Continuous Batching | Reuses decode slots as requests finish and keeps multiple requests active. | Improves aggregate utilization, not single-request latency. The fixed serving snapshot below reports both wall-clock output and timed decode throughput. |
| 3.2 Chunked Prefill | Bounds how long a prompt can delay active decoders. | Primarily improves fairness and time-to-next-token. Smaller chunks add launches and may reduce aggregate throughput; report both latency and throughput. |
| 3.3 Paged Attention, Part 1 | Allocates a paged KV cache with fixed-size reusable pages and separates logical context from physical storage. | Per-layer pools grow geometrically and share storage across requests in that layer. This improves capacity, reuse, removal, and fragmentation without serializing every layer through one backing tensor. |
| 3.4 Paged Attention, Part 2 | Reads K/V through page tables without rebuilding dense per-request K/V. | The completed single-request Week 3 checkpoint reached 207.40 decode tok/s. With four active slots, the paged serving path reached a 327.62 tok/s median during timed decode sections. |
| 3.5 Paged FlashAttention | Tiles the direct Day 4 page walk with Week 2 SIMD-matrix fragments and online softmax. | Its opportunity is long prefill; decode retains the Day 4 vector schedule. The completed checkpoint reached 1,859.97 prefill tok/s at the short course shape. |
| 3.6 Optional MoE | Routes tokens through selected experts and supports sparse Qwen3 variants. | Expands model coverage and can reduce active parameter work, but the course has no controlled dense-versus-MoE speed claim. |
| 3.7 Optional Serving Lab | Varies chunk size, batch size, page size, request mix, and dense-versus-paged policy without adding a model kernel. | Report latency, throughput, memory, fairness, and allocator behavior together. |
| Optional Speculative Decoding | Drafts several tokens and verifies them with the target model. | Work in progress. Speed depends on acceptance rate and cache-rewind cost; no result should be claimed yet. |

### Current Isolated Operator Snapshot

The operator runner uses synchronized median latency over 50 iterations after
10 warmups at the Qwen3-0.6B shapes. Lower latency is better. The speedup column
compares the course kernel with its readable or vanilla course implementation,
not with MLX.

| Operator | Readable/vanilla µs | Course µs | MLX µs | Course speedup |
|---|---:|---:|---:|---:|
| Quantized embedding | 150.7 | 177.4 | 137.6 | 0.85x |
| Q projection | 190.8 | 135.1 | 106.8 | 1.41x |
| K projection | 154.3 | 109.7 | 106.5 | 1.41x |
| V projection | 156.5 | 112.9 | 109.7 | 1.39x |
| O projection | 195.8 | 123.9 | 109.6 | 1.58x |
| Gate projection | 229.2 | 124.1 | 113.8 | 1.85x |
| Up projection | 230.1 | 123.7 | 114.6 | 1.86x |
| Down projection | 235.3 | 126.1 | 120.6 | 1.87x |
| Tied LM head | 4,547.7 | 439.2 | 427.7 | 10.35x |
| Prefill Q matmul | 729.6 | 355.7 | 232.7 | 2.05x |
| RMSNorm | 167.9 | 128.2 | 126.3 | 1.31x |
| RoPE | 168.9 | 111.5 | 108.6 | 1.52x |
| SwiGLU | 147.5 | 106.6 | 108.1 | 1.38x |
| Decode attention | 138.8 | 126.6 | 107.6 | 1.10x |

The direct embedding kernel loses this isolated one-token comparison, so it
should not be described as a standalone speedup. Day 6's end-to-end prefill
gain comes from the SIMD-matrix products and last-token output boundary. The
tied output head remains the largest isolated matvec win because it reads the
widest packed matrix.

### Current Four-Slot Serving Snapshot

This serving comparison fixes all 16 requests at 128 prompt and 65 output
tokens, uses a chunk size of 128, warms up once, alternates paged and
dense-gather processes, and reports the median of three runs per path.

| Week 3 path | Output tok/s | Total tok/s | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| Day 4 dense-gather ablation | 175.02 | 519.66 | 2,423.55 | 207.64 |
| Completed paged path | 248.50 | 737.87 | 2,315.78 | 327.62 |

Paging is 4.4% slower during prefill in this workload, but removes enough
repacking during decode to raise timed decode throughput by 57.8% and
wall-clock output throughput by 42.0%. This is a serving-system result, not a
claim that indirect page reads beat contiguous reads in every isolated kernel.

## Week 4: Measure Application Quality Separately

| Chapter | Appropriate metric |
|---|---|
| 4.1 Agent Loop | Valid-action rate, steps, generated tokens, and wall-clock latency. The loop adds work; it does not improve model tok/s. |
| 4.2 Tools | Successful bounded reads, exact-edit success, command failures, and observation size. |
| 4.3 Safety and Validation | Rejected unsafe actions, false rejections, mutation scope, and post-edit test results. |
| 4.4 Sessions | Resume fidelity, serialized state size, and time to restore useful context. |
| 4.5 Compaction | Token reduction, retained-task-state accuracy, and task completion before and after compaction. |
| 4.6 Control and Recovery | Steering latency, interrupt latency, and successful checkpoint/undo recovery. |
| 4.7 Evaluation | Held-out task-completion rate, test pass rate, steps, tokens, and end-to-end latency. |

The Week 4 chapters remain works in progress. Their success criterion is useful,
safe application behavior on top of the Week 2/3 inference interfaces, not a
higher isolated token rate.

{{#include copyright.md}}
