# Tiny-LLM final-main benchmark and course-claim ledger

## Outcome

The current published harnesses are sufficient; no harness change is needed.
All eight raw files bind to merged main
`18aec8503929d80c986324578068ecac2463c2ac`, tree
`f88ea559a1deea6970160e68acfb1838ef815473`, with a tracked-clean flag and the
same host. The focused Week 2 Days 6–7 and Week 3 Days 2/4/5 reference gates are
green: **74 passed, 3 model-availability skips**. Every stored median recomputes
exactly from its raw samples.

The old Week 2 stretch statement survives, but only with its actual scope:
Qwen3-4B, 128 prompt tokens, 129 output length, last-row logits, two warmups,
and four balanced fresh-process samples. Final Week 2 Split-K reaches **88.2%
of full-MLX prefill and 87.0% of full-MLX decode throughput**. The four paired
sample ratios span 87.2–88.9% for prefill and 82.3–92.3% for decode, so the
median is not carried by one favorable process. Do not generalize the 80%
statement to Qwen3-0.6B, 2K/8K, other GPUs, or Week 3.

The later course patch must also correct an ownership error. Week 2 remains the
course-owned zero-`mlx::steel` loader/direct-`simdgroup_matrix` path. Canonical
dense Week 3 and all three Week 3 serving variants now use the explicit MLX
quantized-projection seam, while cache, attention, paging, batching, and
scheduling remain course-owned. Full MLX is a different baseline and appears
only where the runner explicitly names an `mlx` variant. Task #360 is the
causal seam ablation; this report's final-main runs are representative absolute
benchmarks. They answer different questions and must not be merged into one
old/new table.

## Identity and method

- Host: Mac mini `Mac16,11`, Apple M4 Pro, 20-core GPU, 64 GB RAM,
  macOS 26.5.2 arm64.
- Software: Python 3.12.13, MLX 0.32.0, mlx-lm 0.31.3, NumPy 2.2.6,
  Xcode 26.6, Metal toolchain 32023.883.
- Models: `qwen3-4b` resolves to `Qwen/Qwen3-4B-MLX-4bit`;
  `qwen3-0.6b` resolves to `Qwen/Qwen3-0.6B-MLX-4bit`. The operator artifact
  additionally records the 4B model shape, W4/group-128 quantization, and all
  software/toolchain identities. Its runner has no offline flag; it resolved
  the already-local snapshot and transferred zero model bytes.
- Static and serving throughput: median of four fresh processes with symmetric
  forward/reverse variant order and the documented cooldown. Warmups and MLX
  lazy evaluation are synchronized.
- Week 2 operator sweep: two forward/reverse context passes in one model-loaded
  process; each case has 12 warmups and 60 synchronized calls with every
  implementation order rotated. The stored summary is the harness-defined
  pooled median across the two equal-size passes.
- Week 3 attention operator: median of four balanced fresh-process medians;
  each process has five warmups and 60 synchronized calls.
- Request traces: chunked prefill SHA-256
  `5c04d91264d8da063b8f6d7b9a2dbabbca3b4a810faccd82f71fdbb2ff4e2e89`;
  serving SHA-256
  `59edc851c5baec49e41a65de2ed239533b8c1974fbe407e1063ddc1f41502ed5`.

## Denominator and ownership ledger

| Evidence row | Projections | Cache / attention / paging / scheduler | What it can establish |
|---|---|---|---|
| Week 2 SIMD or Split-K | Course-owned zero-Steel W4 kernels, course-owned loader and direct Metal SIMD-matrix helper | Course-owned Week 2 dense cache and operators | Week 2 course implementation versus the explicitly paired full-MLX row. |
| Week 3 chunked, dense serving, paged-gather serving, direct-paged serving, or Week 3 8K | Explicit MLX quantized-projection seam | Course-owned cache, attention, paging, batching, and scheduling | Representative cumulative Week 3 behavior. It does not isolate the seam. |
| Full `mlx` row | Full MLX model/operator, as named by that runner | Full MLX | External denominator. It is not the same as a Week 3 course row that only uses MLX projections. |
| Task #360 seam versus inherited | MLX quantized projections versus the inherited Week 2 course projection path | Identical course-owned Week 3 mechanisms on one measured source tree | Causal projection-seam effect. Do not substitute its absolute 0.6B numbers for the final-main 4B serving table. |

Final-main required-solution source has no `mlx::steel` include or Steel
`BlockLoader`/`BlockMMA` symbol. `src/extensions_ref/src/cooperative_matrix.h` supplies
the course-owned `CooperativeTileLoader` and `CooperativeBlockMMA` over Metal
`simdgroup_matrix`; Week 2 quantized matmul and Week 3 paged attention include
that header.

## Final-main representative results

### Week 2 cumulative Qwen3-4B acceptance ladder

Workload: input 128, output 129, last-row logits, two warmups, four balanced
fresh processes. Throughput is tokens/s.

| Cumulative checkpoint | Prefill | Output | Decode | Prefill / MLX | Output / MLX | Decode / MLX |
|---|---:|---:|---:|---:|---:|---:|
| Dense KV cache | 706.65 | 21.25 | 21.73 | 88.1% | 30.5% | 28.7% |
| Quantized matvec | 104.82 | 36.77 | 55.96 | 13.1% | 52.7% | 73.9% |
| Fast RMSNorm | 104.88 | 39.93 | 63.70 | 13.1% | 57.3% | 84.2% |
| + Fast RoPE | 105.37 | 40.97 | 66.20 | 13.1% | 58.7% | 87.5% |
| + Fused SwiGLU | 105.84 | 41.65 | 67.83 | 13.2% | 59.7% | 89.6% |
| Decode attention | 105.90 | 42.89 | 71.18 | 13.2% | 61.5% | 94.1% |
| SIMD-matrix prefill | 706.50 | 61.05 | 66.28 | 88.0% | 87.5% | 87.6% |
| Split-K prefill | 707.41 | 60.67 | 65.83 | **88.2%** | **87.0%** | **87.0%** |
| Full MLX 0.32.0 | 802.50 | 69.75 | 75.68 | 100% | 100% | 100% |

The acceptance effect of Split-K itself is neutral: +0.13% prefill, -0.61%
output, and -0.69% decode versus the SIMD checkpoint. The final checkpoint is
still well above 80% of MLX in both required phases because that acceptance is
for the complete Week 2 path, not because Split-K speeds up M=128.

### Week 2 short and long controls

| Input | Path | Prefill tok/s | Output tok/s | Decode tok/s | Prefill / MLX |
|---:|---|---:|---:|---:|---:|
| 32 | SIMD | 537.92 | 62.16 | 67.97 | 76.6% |
| 32 | Split-K | 599.81 | 62.66 | 67.62 | 85.4% |
| 32 | Full MLX | 702.61 | 71.55 | 77.11 | 100% |
| 2,048 | SIMD | 551.48 | 18.84 | 40.84 | 75.9% |
| 2,048 | Split-K | 547.73 | 18.80 | 40.70 | 75.3% |
| 2,048 | Full MLX | 727.04 | 27.59 | 67.72 | 100% |

At 32 tokens, Split-K adds **11.5%** complete-model prefill versus SIMD while
decode/output remain neutral. At 2,048 tokens it is **0.7% lower**, again
neutral; the policy uses the unsplit path once the result grid is occupied.
The long control is not an 80%-of-MLX acceptance point.

### Week 2 Qwen3-4B projection operator sweep

The following are the harness-reported pooled medians in microseconds (lower is
better). Absolute M=32 times shifted substantially between the first and last
context pass, so the replacement prose should use the paired direction check,
not a categorical ranking from one pooled percentage.

| M | Projection | SIMD | Split-K | MLX |
|---:|---|---:|---:|---:|
| 32 | Q | 566.1 | 513.1 | 506.0 |
| 32 | K | 270.9 | 258.1 | 235.7 |
| 32 | V | 243.1 | 191.4 | 191.7 |
| 32 | O | 287.5 | 275.7 | 261.3 |
| 32 | gate | 443.8 | 448.2 | 417.5 |
| 32 | up | 446.3 | 443.0 | 416.0 |
| 32 | down | 493.8 | 448.5 | 417.9 |
| 128 | Q | 662.2 | 661.6 | 623.4 |
| 128 | K | 284.5 | 282.4 | 265.4 |
| 128 | V | 281.4 | 280.8 | 265.3 |
| 128 | O | 643.4 | 634.6 | 587.7 |
| 128 | gate | 1,312.2 | 1,298.4 | 1,235.6 |
| 128 | up | 1,266.9 | 1,258.6 | 1,179.0 |
| 128 | down | 1,381.4 | 1,363.5 | 1,277.8 |
| 2,048 | Q | 7,329.5 | 7,486.0 | 6,872.2 |
| 2,048 | K | 2,060.1 | 2,058.0 | 1,902.7 |
| 2,048 | V | 2,059.7 | 2,056.2 | 1,903.0 |
| 2,048 | O | 7,634.7 | 7,651.6 | 6,906.9 |
| 2,048 | gate | 18,038.4 | 18,100.4 | 16,889.7 |
| 2,048 | up | 18,593.4 | 18,386.1 | 16,894.9 |
| 2,048 | down | 19,384.4 | 19,361.8 | 17,421.1 |

Repeat-level M=32 direction is the causal guard. Split-K improves K by
29.2%/14.9%, V by 23.6%/11.3%, O by 3.9%/4.9%, and down by 11.1%/8.8% in the
two context positions. Gate/up are neutral and Q reverses direction (-4.5%,
+1.5%). At M=128 changes are small/mixed and the complete-model result is
neutral. At M=2,048 the split policy falls back to the unsplit schedule; paired
operator differences are within about 1% except pooled Q, whose 2.1% difference
is another drift warning. The course should no longer say that all occupied
controls are within 3.1% of MLX: current SIMD latency is roughly 5–10% above MLX
at M=128 and 8–11% above it at M=2,048.

### Week 3 Qwen3-0.6B chunked-prefill tradeoff

All rows use the final-main Week 3 projection seam and identical course-owned
scheduler/cache/attention code; this table varies only the prefill budget.

| Prefill budget | Output tok/s | Prefill tok/s | Decode tok/s | Requests/s | Decode step p95 | Decode gap p95 / max |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 105.23 | 2,549.62 | 181.77 | 3.288 | 15.82 ms | 30.01 / 52.62 ms |
| 128 | 153.82 | 4,215.12 | 242.23 | 4.807 | 17.79 ms | 45.36 / 53.76 ms |
| 512 | 170.46 | 4,769.14 | 262.01 | 5.327 | 17.11 ms | 73.56 / 119.90 ms |

The 128-token budget remains the course compromise for this trace: versus the
full-prompt 512 control, it gives up 9.8% output throughput while reducing the
p95 completion gap by 38.3% and the maximum by 55.2%. This is not a universal
chunk-size threshold.

### Week 3 Qwen3-4B direct-paged decode operator

| Cached context | Dense + gather | Direct paged | MLX fused | Direct versus dense |
|---:|---:|---:|---:|---:|
| 128 | 201.26 us | 228.58 us | 188.79 us | 13.6% slower |
| 1,024 | 468.39 us | 299.14 us | 250.04 us | 36.1% faster |

Direct and MLX outputs match the dense BF16 equation within 0.00439453125 at
S=128 and 0.001953125 at S=1,024. The qualitative crossover survives; the old
1.9%/40.7% values do not. This operator benchmark contains no model projection,
so it directly isolates the attention paths.

### Week 3 Qwen3-4B continuous serving

All three course paths below use the explicit MLX quantized-projection seam.
They differ in KV representation and attention path, not projection backend.

| Storage / attention path | Prefill tok/s | Output tok/s | Decode tok/s | Requests/s | Peak KV | Avoidable KV copy |
|---|---:|---:|---:|---:|---:|---:|
| Dense growth and reconstruction | 711.18 | 35.23 | 57.59 | 0.469 | 1,096 MiB | 209,532 MiB |
| Paged storage + dense gather | 725.46 | 41.64 | 78.53 | 0.555 | not a total peak | 103,445 MiB |
| Direct paged attention | 672.68 | 46.36 | 105.01 | 0.618 | 576 MiB | 504 MiB |

The compatibility row's 603 MiB page-pool counter excludes its temporary dense
staging allocation, so it must not be printed as a total peak. Relative to dense
serving, direct paging is 5.4% lower on prefill, 31.6% higher on output/request
throughput, 82.3% higher on decode, 47.4% lower on measured peak KV storage, and
99.76% lower on avoidable logical copy volume. Relative to paged+gather it is
7.3% lower on prefill, 11.3% higher on output/request throughput, 33.7% higher
on decode, and 99.51% lower on copy volume.

| Path | Decode step median / p95 / max | Completion gap median / p95 / max |
|---|---:|---:|
| Dense | 51.03 / 84.49 / 124.52 ms | 53.16 / 248.30 / 309.74 ms |
| Paged + gather | 39.80 / 52.79 / 80.09 ms | 41.82 / 225.64 / 261.38 ms |
| Direct paged | 28.97 / 36.78 / 63.04 ms | 30.16 / 222.18 / 239.49 ms |

These are cumulative system results. They do not isolate the Day 5 prefill
kernel from paging, direct decode, allocation, or scheduling, and they must not
credit the MLX projection seam to paged attention.

### Qwen3-4B 8K static diagnostic

| Static checkpoint | Prefill tok/s | Decode tok/s | Prefill / full MLX | Decode / full MLX |
|---|---:|---:|---:|---:|
| Week 2 course-owned projections | 323.96 | 17.73 | 50.6% | 62.5% |
| Week 3 seam + course paged path | 463.69 | 27.42 | 72.5% | 96.7% |
| Full MLX | 639.73 | 28.37 | 100% | 100% |

Week 3 prefill is 43.1% above Week 2 and reaches 72.5% of full MLX. This remains
a static long-query diagnostic: it does not measure request turnover, page
reuse, admission capacity, or the projection seam causally. Its one-token
decode row exercises the Day 4 vector schedule, not the Day 5 tiled prefill
kernel.

## Task #360 causal projection-seam evidence

The checked task #360 ledger verifies all 20 files. It measures source
`170211be3503c0ec0b1fa75bbb3b0c23a86bd3ac`, tree
`25e588de248692b711e04014ba43e85326d21393`, with identical course mechanisms
on both sides and a benchmark-only inherited-projection control.

| Causal comparison | Seam effect versus inherited Week 2 projections |
|---|---:|
| Chunked prefill, step 512 | +10.64% prefill; +11.91% output |
| Chunked prefill, step 128 | +11.74% prefill; +11.76% output |
| Dense Day 3 | +12.17% prefill; +16.82% output; +18.86% decode |
| Serving | +7.72% prefill; +9.42% output; +13.02% decode |

Full MLX remains 17.83% faster than the Day 3 seam on prefill (equivalently,
the seam is 15.13% below full MLX), because the seam changes projections only.
Use these percentages to explain why Week 3 changed projection ownership. Use
the final-main tables above for the present absolute numbers. Do not compare
task #360's 0.6B absolute serving numbers to the final-main 4B serving trace.

## Seven-file old-to-new map

| File and current surface | Required later patch |
|---|---|
| `week2-overview.md` lines 44, 61–68 | Keep the optional 80% target, but name the exact Qwen3-4B 128/129 acceptance workload and current 88.2%/87.0% result. Keep the statement that Week 2 implements its own learned operators; add that the required solution uses a course-owned cooperative loader and direct Metal SIMD matrices, not Steel. |
| `week2-overview.md` lines 118–121 | Replace “Week 3 imports these Week 2 interfaces” prose that implies projection-schedule inheritance. Week 3 keeps the interfaces and course-owned cache/attention/paging/scheduler, but explicitly selects MLX quantized projections for dense Week 3 and the Week 3 scheduler factory. |
| `week2-06-simd-matrix-prefill.md` lines 65–69 | Replace permission to use Steel `BlockLoader`/`BlockMMA` with the actual `cooperative_matrix.h` boundary: `CooperativeTileLoader`, direct `simdgroup_matrix` fragments, and `CooperativeBlockMMA`. The solution still owns W4 unpack/dequantization, layout, dispatch, split policy, and reduction. |
| `week2-06-simd-matrix-prefill.md` lines 139–195 | Keep the commands and causal long/short method. Point result links to the new corpus. Replace “long controls approach MLX” with the measured 8–11% long operator gap and say the control establishes that Split-K is neutral once the base grid is occupied, not parity with MLX. |
| `week2-07-split-k-prefill.md` lines 7–10 and 147–190 | Replace the operator/end-to-end values. The supported story is: M=32 K/V/O/down improve in both balanced positions, Q is mixed, gate/up are neutral; complete-model prefill improves 11.5%; M=128 and M=2,048 are neutral. Retain the 80% acceptance requirement with the exact 88.2%/87.0% result. |
| `week2-07-split-k-prefill.md` lines 192–206 | Replace “Week 3 inherits these projection schedules.” Week 3 inherits interfaces and mechanisms but selects MLX quantized projections. Paging still gets no credit for projection gains; task #360 supplies the separate causal seam result. |
| `week3-02-chunked-prefill.md` lines 89–101 | Replace the 32/128/512 table with 105.23/153.82/170.46 output tok/s and 30.01/45.36/73.56 ms gap p95 (plus 52.62/53.76/119.90 ms max). State that all rows use the same Week 3 MLX-projection seam and compare scheduler budgets only. |
| `week3-04-paged-attention-part2.md` lines 443–453 | Replace the attention table with 201.26/228.58/188.79 us at S=128 and 468.39/299.14/250.04 us at S=1,024. Replace 1.9% slower / 40.7% faster with 13.6% slower / 36.1% faster. Preserve the operator-only scope. |
| `week3-04-paged-attention-part2.md` lines 525–532 | Replace serving values with 672.68 prefill, 46.36 output, 105.01 decode tok/s, 0.618 requests/s, step 28.97/36.78/63.04 ms, and gap 30.16/222.18/239.49 ms. Add that all three paths share the MLX projection seam. |
| `week3-05-flash-attention.md` lines 95–102 and 116 | Replace Steel loader wording with the course-owned cooperative loader and direct `simdgroup_matrix` fragments reused from Week 2. Page translation, cross-page fallback, masking, online softmax, and dispatch remain course-owned. |
| `week3-05-flash-attention.md` lines 183–229 | Replace serving and 8K tables with the values above. The supported 8K statement is +43.1% over Week 2 and 72.5% of full MLX prefill. Preserve the explicit nonclaim that serving is cumulative and 8K is not serving/admission evidence. |
| `appendix-performance.md` lines 178–206 | Replace the complete 128-token ladder with the current nine-row table. The old “historical Day 5” caveat is obsolete because the current ladder measured the current guard. Day 2 may repeat the Day 1 row because it remains the benchmark-method checkpoint. |
| `appendix-performance.md` lines 377–456 | Replace Day 6/7 operator and 32/128/2,048 complete-model tables. Narrow the occupancy claim to the repeat-stable K/V/O/down M=32 results; disclose Q mixed and gate/up neutral. Replace 95.4%/84.4% with 88.2%/87.0% at acceptance. |
| `appendix-performance.md` lines 458–561 | Replace chunked, attention, serving, latency, and 8K values. Insert the ownership/denominator ledger before interpreting them. Add a clearly separate task #360 causal seam table; do not splice its percentages into the final-main absolute progression. |

## Wording boundaries

Supported:

- “On this M4 Pro and the fixed Qwen3-4B 128/129 acceptance workload, the
  final Week 2 checkpoint reached 88.2% of full-MLX prefill and 87.0% of
  full-MLX decode throughput.”
- “At 32 prompt rows, Split-K improved complete-model prefill by 11.5%; at 128
  and 2,048 rows it was neutral.”
- “The explicit Week 3 MLX projection seam improved the task #360 inherited-path
  ablation by 7.7–18.9% depending on workload and phase, while course cache,
  attention, paging, and scheduling stayed unchanged.”
- “Direct paged attention trades short-context operator latency and serving
  prefill throughput for lower copy volume, lower measured peak KV storage,
  and higher decode/output throughput on this fixed trace.”

Not supported:

- a universal 80% threshold, a Qwen3-0.6B-to-4B extrapolation, or any claim that
  both phases exceed 80% at 2K/8K;
- “Week 3 inherits Week 2 projection schedules” or “Week 3 is fully
  course-owned” without naming the MLX projection seam;
- crediting paged attention, FlashAttention, or scheduling with task #360's
  projection gain;
- calling a Week 3 course row “full MLX,” calling the serving compatibility
  path's page-pool bytes a total peak, or claiming a short-chunk prefill win;
- a universal Split-K crossover, categorical operator ranking under the M=32
  drift, production throughput, admission-capacity proof, or long-context
  support from the 8K static diagnostic.

## Reproduction commands

The raw JSON stores the complete normalized configuration and execution order.
The campaign corresponds to these current public runners (all paths abbreviated
below point to the raw corpus):

```bash
pdm run bench-week2-progression --offline --solution ref --suite week2 \
  --model qwen3-4b --input-len 128 --output-len 129 \
  --prefill-logits last --warmup 2 --repeats 4 --cooldown-seconds 1 \
  --json-output week2-128-final-main.json

pdm run bench-week2-progression --offline --solution ref --suite week2 \
  --variant week2-simd-matmul --variant week2-split-k --variant mlx \
  --model qwen3-4b --input-len 32 --output-len 33 \
  --prefill-logits last --warmup 2 --repeats 4 --cooldown-seconds 1 \
  --json-output week2-32-final-main.json

pdm run bench-week2-progression --offline --solution ref --suite week2 \
  --variant week2-simd-matmul --variant week2-split-k --variant mlx \
  --model qwen3-4b --input-len 2048 --output-len 129 \
  --prefill-logits last --warmup 2 --repeats 4 --cooldown-seconds 1 \
  --json-output week2-2048-final-main.json

pdm run bench-week2-operators --solution tiny_llm_ref --model qwen3-4b \
  --section prefill-projections --context 32 --context 128 --context 2048 \
  --context-repeats 2 --prefill-projection q --prefill-projection k \
  --prefill-projection v --prefill-projection o --prefill-projection gate \
  --prefill-projection up --prefill-projection down --include-split-k \
  --warmup 12 --iterations 60 \
  --json-output week2-prefill-operators-final-main.json

pdm run bench-chunked-prefill --offline --model qwen3-0.6b \
  --prefill-steps 32 128 512 --num-seqs 8 --batch-size 4 \
  --min-input-len 64 --max-input-len 512 \
  --min-output-len 32 --max-output-len 32 \
  --warmup 1 --repeats 4 --cooldown-seconds 1 \
  --json-output week3-chunked-prefill-final-main.json

pdm run bench-week3-attention --offline --contexts 128 1024 \
  --page-size 128 --warmup 5 --iterations 60 --repeats 4 \
  --cooldown-seconds 1 --json-output week3-attention-final-main.json

pdm run bench-serving-progression --offline --repeats 4 --model qwen3-4b \
  --num-seqs 16 --batch-size 4 --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 --prefill-step 128 \
  --warmup 1 --cooldown-seconds 1 \
  --json-output week3-serving-final-main.json

pdm run bench-course-progression --offline --solution ref --suite course \
  --variant week2 --variant week3 --variant mlx --model qwen3-4b \
  --input-len 8192 --output-len 2 --prefill-logits last \
  --warmup 1 --repeats 4 --cooldown-seconds 1 \
  --json-output week3-8k-final-main.json
```

The raw archive manifest lists every file SHA-256. The repository remained
tracked-clean at the exact merged head throughout; no source, book, starter,
test, PR, release, or deployment state changed.
