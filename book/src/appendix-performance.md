# 🚧 Appendix: Performance by Chapter

> 🚧 This newly introduced appendix is a work in progress.

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

## End-to-End Checkpoints and Reverse Ablations

The following post-restructure snapshot used Qwen3-0.6B on an Apple M4 Pro
with a 20-core GPU and 64 GB of memory. Each row used a 128-token prompt, 65
generated tokens, and two complete warmups. Every row is a three-process
median from the course progression runner. The Week 3 rows are reverse
ablations of the completed paged stack, not the order in which the chapters
introduce the features.

| Checkpoint | Prefill tok/s | Versus Week 1 | Gap to MLX | Decode tok/s | Versus Week 1 | Gap to MLX |
|---|---:|---:|---:|---:|---:|---:|
| Week 1 readable model | 3,340.46 | baseline | 24.2% slower | 19.48 | baseline | 94.0% slower |
| Week 2 decode checkpoint | 778.71 | 76.7% slower | 82.3% slower | 239.48 | 12.29x | 26.5% slower |
| Week 3 paged stack (Flash off) | 655.47 | 80.4% slower | 85.1% slower | 37.82 | 1.94x | 88.4% slower |
| Week 3 paged stack + FlashAttention | 742.93 | 77.8% slower | 83.1% slower | 37.67 | 1.93x | 88.4% slower |
| Week 3 paged stack + performance lab | 1,118.09 | 66.5% slower | 74.6% slower | 37.84 | 1.94x | 88.4% slower |
| Week 3 paged stack + FlashAttention + performance lab | 1,420.27 | 57.5% slower | 67.8% slower | 37.70 | 1.94x | 88.4% slower |
| MLX | 4,407.38 | 1.32x | baseline | 325.70 | 16.72x | baseline |

Week 1 prefill is fast because it materializes dense 16-bit weights and uses
efficient dense matrix multiplication. Week 2 deliberately optimizes the
memory-bound one-token decode shape first; its readable scalar quantized
prefill path therefore regresses. Week 3's optional SIMD-matrix lab recovers
part of that prefill gap. The current paged-attention kernel dominates Week 3
decode and is a teaching implementation, not a claim of production speed. The
runner releases every request cache, including warmups, so paged checkpoints
reuse allocator capacity rather than timing pool growth left by an earlier run.

## Week 1: Establish the Readable Baseline

| Chapter | Performance or usability effect | Comparison |
|---|---|---|
| 1.1 Attention | Materializes scores and probabilities so the algorithm is inspectable. | No speed claim; this is the readable attention baseline. |
| 1.2 RoPE | Builds positions, sine, and cosine with array operations. | No isolated checkpoint gain; Week 2 later fuses this work. |
| 1.3 GQA | Shares K/V heads, reducing KV storage versus full multi-head attention. | The Qwen3 architecture already fixes the head ratio, so the course has no MHA-to-GQA timing ablation. |
| 1.4 RMSNorm and MLP | Uses readable array expressions and exposes intermediate tensors. | No speed claim; Week 2 measures fused replacements against these equations. |
| 1.5 Qwen3 Model | Produces the first complete model and the baseline prefill graph. | The completed Week 1 checkpoint measured about 3,350 prefill tok/s, 24.1% below MLX. |
| 1.6 Generation | Recomputes the growing prefix for every output token. | About 19.51 decode tok/s at context 128, roughly 94.0% below MLX. |
| 1.7 Sampling | Adds temperature/top-p/top-k policy after logits. | Not a model-throughput optimization; deterministic argmax is used by the course benchmark. |

## Week 2: Build the Fast Decode Checkpoint

| Chapter | What changes | Measured contribution and MLX comparison |
|---|---|---|
| 2.1 KV Cache | Prefills once, stores K/V, and processes one new query during decode while retaining the readable Week 1 operators. | 19.51 to 101.80 tok/s: 5.22x Week 1 and 68.9% below MLX. This algorithmic change creates the decode-shaped workload optimized later. |
| 2.2 Benchmark | Synchronizes lazy work, separates prefill from decode, and records the cached baseline and MLX denominator. | 0% direct speedup. It makes every later cumulative percentage reproducible. |
| 2.3 Quantized Matvec | Keeps weights packed and maps one-token products across SIMD groups, then integrates the kernel into every projection. | 101.80 to 134.24 tok/s: +31.9% over the preceding cached checkpoint, 6.88x Week 1, and 59.1% below MLX. Ordinary kernels were 1.4-1.9x and the vocabulary head 10.5x faster than the scalar quantized oracle. |
| 2.4 Decode Attention | Replaces the exact Week 1 float32 score/softmax/value composition with a course-owned online-softmax kernel for its measured short-context range. | 134.24 to 143.42 tok/s: +6.8% over quantized matvec, 7.35x Week 1, and 56.3% below MLX. |
| 2.5 Fast Kernels | Integrates RMSNorm, then RoPE, then SwiGLU as three separately measured checkpoints. | RMSNorm: 143.42 → 193.18 (+34.7%). RoPE: 193.18 → 222.51 (+15.2%). SwiGLU: 222.51 → 242.67 (+9.1%). The retained model is 12.44x Week 1 and 26.0% below MLX. |
| Week 2 checkpoint | Combines only changes already integrated and measured in chapter order. | 242.67 tok/s, about 12.44x Week 1 and 26.0% slower than matched MLX. Quantized prefill remains intentionally slower. |

## Week 3: Improve Serving Structure

| Chapter | What improves | Performance reality |
|---|---|---|
| 3.1 Continuous Batching | Reuses decode slots as requests finish and keeps multiple requests active. | Improves aggregate utilization, not single-request latency. In one four-request snapshot, dense batching reached 288.4 aggregate decode tok/s versus 240.4 for one Week 2 request; do not treat different batch sizes as a kernel speedup. |
| 3.2 FlashAttention | Streams K/V tiles with online softmax and bounds scratch memory instead of materializing `L x S`. | As a reverse ablation inside the completed paged stack, enabling it improved prefill from 655.47 to 742.93 tok/s (+13.3%) but remained 83.1% below MLX. Decode did not move. The standalone course kernel is about 9% slower than explicit attention at 4096 tokens and about 3.4x slower than MLX at 2048, while avoiding roughly 1 GiB of scores at the documented 4096-token Qwen3-4B shape. |
| 3.3 Chunked Prefill | Bounds how long a prompt can delay active decoders. | Primarily improves fairness and time-to-next-token. Smaller chunks add launches and may reduce aggregate throughput; report both latency and throughput. |
| 3.4 Paged Attention, Part 1 | Allocates a paged KV cache with fixed-size reusable pages and separates logical context from physical storage. | Improves capacity, reuse, removal, and fragmentation. It is an allocator/usability gain; no raw kernel speedup is claimed. |
| 3.5 Paged Attention, Part 2 | Reads K/V through page tables without rebuilding dense per-request K/V. | The current operator is about 4.4-16.7x slower than MLX across the documented batch/context cases. The end-to-end paged model reached 37.82 decode tok/s, about 88.4% below MLX. |
| 3.6 Optional MoE | Routes tokens through selected experts and supports sparse Qwen3 variants. | Expands model coverage and can reduce active parameter work, but the course has no controlled dense-versus-MoE speed claim. |
| 3.7 Optional Performance Lab | Adds SIMD-matrix quantized prefill, direct quantized embedding, and last-token projection analysis. | Without FlashAttention, it improved the completed paged stack from 655.47 to 1,118.09 prefill tok/s (+70.6%) and remained 74.6% below MLX. With FlashAttention, it improved 742.93 to 1,420.27 prefill tok/s (+91.2%) and remained 67.8% below MLX. Paged decode remained about 37.8 tok/s. |
| 3.8 Optional Speculative Decoding | Drafts several tokens and verifies them with the target model. | Work in progress. Speed depends on acceptance rate and cache-rewind cost; no result should be claimed yet. |

Paged attention is currently slower even under batching. In one matched
four-request snapshot, the dense compatibility path reached 288.4 aggregate
decode tok/s while the Week 3 paged path reached 26.6 tok/s. Paging still
teaches the correct ownership and memory-management structure, but the page
walker needs a decode-specific schedule before it becomes a performance win.

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
