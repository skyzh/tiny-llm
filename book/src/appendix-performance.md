# 🚧 Appendix: Performance Evidence and Decision Ledger

This appendix binds Week 2 claims to the accepted frozen successor. It separates
correctness, routing, component, product, historical, and unavailable evidence.
Numbers in different categories or with different denominators are not additive.

## Bound Successor and Measurement Package

The executable successor is source head
`d9eba2b632dd2d65e04770762c8b6f3ef6b5bad8`, tree
`b6cdcdc5c85082d8a0a9d5303945ac6f4ab261cb`. Its public Week 2 checkpoints
are, in order:

`kv-cache` → `capacity-cache` → `quantized-matvec` → `simd-matmul` →
`rmsnorm` → `rope` → `swiglu` → `tiled-prefill` → `selected`.

The accepted performance package was measured on its exact predecessor
mechanisms using Qwen3-4B-MLX-4bit snapshot
`52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495`, Python 3.12.13, MLX 0.32.0,
and MLX-LM 0.31.3. Four product rows completed with six balanced paired fresh
processes per arm and two excluded warmups. Two declared rows are unavailable.

These are supplied accepted results. Legacy benchmark/profiler defaults in this
source tree predate the successor and are not learner commands for the route
above. Current chapters show only commands accepted by the frozen public
interface.

## Evidence Vocabulary

| Label | What it supports | What it cannot support alone |
|---|---|---|
| Correctness | declared output/mask/state contract | optimized path ran or was faster |
| Routing | dispatch/copy counter selected the intended path | lower latency |
| Component | one operator improved at one fixed shape | complete-request gain |
| Product | complete matched request changed | isolated causal attribution |
| Control | denominator for the same workload | a result for another shape |
| Historical | earlier exact-mechanism tradeoff | fresh final-head behavior |
| Unavailable | no accepted value exists | zero, failure, or an estimate |

Source geometry—thread counts, tile sizes, and static storage—is not a hardware
counter. No occupancy, bandwidth, or cache claim is inferred from it.

## Selected Mechanisms and Fallbacks

| Mechanism | Required invariant | Readable/control path | Main interpretation trap |
|---|---|---|---|
| Request-bounded KV capacity | logical prefix is `:offset`; overflow is transactional; rewind/reset never expose abandoned slots | concatenating dense cache at `kv-cache` | treating zero prefix-copy counters as a product speedup or universal peak-memory reduction |
| Register-cached RMSNorm | FP32 reduction, BF16 model contract, supported widths through 4096 | fixed-width native fallback above 4096; readable RMSNorm before the checkpoint | adding component percentages to request gains |
| Tiled dense prefill | BF16 D128, BQ32/BK16, correct GQA/masks/tails, finite-zero fully masked rows | readable grouped attention for short or ineligible queries | extending a prefill result to single-query decode |
| Packed W4 | W4/group-128 layout and course-owned unpack/matmul | readable W4 Metal control and earlier dense checkpoints | equating storage reduction with throughput |
| SIMD matrix prefill | packed equation unchanged; partial tiles guarded | Day 3 matvec for small M; vanilla matrix control | inferring occupancy from a timing change |
| RoPE / SwiGLU | correct offsets and elementwise semantics | readable MLX expressions | treating every cumulative checkpoint as an independently selected mechanism |

A bounded single-kernel decode experiment is not a current Week 2 optimization.
The reduction-splitting projection experiment is also retired. Neither is a
current checkpoint, runnable learner command, or TODO.

## Accepted Component Evidence

| Mechanism / fixed shape | Favorable change | Category | Boundary |
|---|---:|---|---|
| RMSNorm / 1×2048×2560 | +71.36% | component | numerical bound passed |
| RMSNorm / 1×8192×2560 | +83.81% | component | numerical bound passed |
| Tiled prefill / q2048/c2048 | +54.07% | component | BF16 D128, selector passed |
| Tiled prefill / q8192/c8192 | +54.89% | component | BF16 D128, selector passed |
| Capacity append / q1/c2048 | +12.44% | component | zero prefix-copy bytes |
| Capacity append / q1/c8192 | +43.38% | component | zero prefix-copy bytes |

Tiled prefill carries online maximum/sum state and does not allocate a score
workspace. The result does not mean every mask representation is subquadratic:
an explicit additive mask may itself occupy `L × S`.

## Accepted Complete-Request Evidence

`Δ total` is the paired favorable total-latency change from all mechanisms
off. `MLX ratio` is selected-path throughput divided by the matched full-MLX
denominator.

| Prompt/output | All-off total | Selected total | Δ total | MLX ratio | 80% direction |
|---:|---:|---:|---:|---:|---|
| 128/128 | 2.221 s | 1.986 s | +10.849% | ≈0.806 | met |
| 512/128 | 2.953 s | 2.653 s | +10.347% | ≈0.822 | met |
| 2K/16 | 3.903 s | 3.312 s | +15.402% | ≈0.828 | met |
| 2K/128 | 6.563 s | 5.639 s | +14.126% | ≈0.770 | missed |
| 2K/512 | unavailable | unavailable | unavailable | unavailable | no verdict |
| 8K/128 | unavailable | unavailable | unavailable | unavailable | no verdict |

The 2K/512 attempt and its single authorized retry were contaminated by
Time Machine/Spotlight activity and are excluded. The 8K/128 attempt stopped
when the in-run guard observed environmental activity before a result was
appended. No retry, component substitution, or favorable-sample imputation
fills either row.

All arms passed the declared semantic, numerical, cache-offset, and repeat
determinism gates. The complete quality score was 10/12 for every arm, with the
same two shared misses; no selected mechanism regressed a control-pass case.

## Historical Capacity Tradeoff

An earlier exact-mechanism measurement observed **+88.0 MiB / +2.276%** temporal
peak at 2K/512. This is historical evidence, not a final-head product result.
It illustrates the exchange: request-bounded allocation removes repeated
prefix copies but may reserve storage earlier. It does not fill the unavailable
2K/512 product row.

## Fresh Successor Decision Ledger

Start fresh for this exact successor. Do not copy keep/reject language from an
older chapter order.

| Checkpoint | Workload/control | Correctness + routing | Component result | Product result | Decision + falsifier |
|---|---|---|---|---|---|
| `kv-cache` | | | | | |
| `capacity-cache` | | | | | |
| `quantized-matvec` | | | | | |
| `simd-matmul` | | | | | |
| `rmsnorm` | | | | | |
| `rope` | | | | | |
| `swiglu` | | | | | |
| `tiled-prefill` | | | | | |
| `selected` | | | | | |

Write `not measured` or `unavailable` instead of guessing. A completed row
names the exact request, the immediate control, the path witness, and the result
that would reverse the decision.

The later-course evidence below is preserved as a frozen record. Any older
Week 2 label inside its historical denominator table describes that archived
run; it is not a current checkpoint or learner command.

## Week 3 Performance by Chapter

Paging adds indirect K/V reads and is not expected to beat contiguous
attention for one preallocated static request. Week 3 therefore measures a
serving workload with request turnover, incremental unknown-size growth,
chunked admission, dense batch reconstruction, and page reuse:

```bash
pdm run bench-serving-progression --offline --repeats 4 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 \
  --prefill-step 128 --warmup 1 --cooldown-seconds 1 \
  --json-output benchmark_results/task367-final-main/raw/week3-serving-final-main.json
```

A complete warmup compiles the kernels. The runner then synchronizes and resets
every page pool, so the measured paged run starts with zero pages and zero
backing capacity.

### Ownership and denominators

The projection boundary must be fixed before interpreting any Week 3 table:

| Evidence row | Projections | Cache / attention / paging / scheduler | What it establishes |
|---|---|---|---|
| Week 2 SIMD or Split-K | Course-owned zero-Steel W4 kernels, loader, and direct SIMD-matrix helper | Course-owned Week 2 dense cache and operators | Week 2 course implementation versus its explicitly paired full-MLX row. |
| Week 3 course row | Explicit MLX quantized-projection seam | Course-owned cache, attention, paging, batching, and scheduling | Representative cumulative Week 3 behavior; it does not isolate the seam. |
| Full `mlx` row | Full MLX model/operator | Full MLX | External denominator, distinct from the hybrid Week 3 course row. |
| Task #360 seam versus inherited | MLX quantized projections versus inherited Week 2 course projections | Identical course-owned Week 3 mechanisms | Causal projection-seam effect on one measured source tree. |

Task #360 and task #367 answer different questions. The former is a causal
ablation; the latter is representative final-main absolute evidence. Do not
splice one campaign's absolute values into the other or credit its projection
gain to paging, FlashAttention, or scheduling.

The Days 1–2 chunk-size control uses one deterministic Qwen3-0.6B trace with
seed 0, eight 64–512-token prompts, a fixed 32-token output budget, and four
balanced fresh processes. A gap is measured between synchronized decode-call
completions only while a decode request is active. Every row uses the same
Week 3 projection seam and course-owned mechanisms; only the budget changes:

| Prefill budget | Output tok/s | Prefill tok/s | Decode tok/s | Requests/s | Decode step p95 | Decode gap p95 / max |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 105.23 | 2,549.62 | 181.77 | 3.288 | 15.82 ms | 30.01 / 52.62 ms |
| 128 | 153.82 | 4,215.12 | 242.23 | 4.807 | 17.79 ms | 45.36 / 53.76 ms |
| 512 | 170.46 | 4,769.14 | 262.01 | 5.327 | 17.11 ms | 73.56 / 119.90 ms |

Because 512 covers every prompt in this trace, that row is the full-prompt Day
1 control. Relative to it, 128 gives up 9.8% output throughput while reducing
the p95 completion gap by 38.3% and the maximum by 55.2%. The course chooses
128 for this trace, not as a universal chunk-size threshold.

The Day 4 operator control uses `B=1`, `Hq=32`, `Hkv=8`, `L=1`, `D=128`, BF16,
and 128-token pages. Each row is the median of four balanced fresh-process
medians, each containing 60 synchronized calls after five warmups:

| Context | Dense + gather | Direct paged | MLX fused |
|---:|---:|---:|---:|
| 128 | 201.26 us | 228.58 us | 188.79 us |
| 1,024 | 468.39 us | 299.14 us | 250.04 us |

The direct operator is 13.6% slower than dense-plus-gather at 128 tokens and
36.1% faster at 1,024 tokens. MLX remains faster at both shapes. Outputs match
the dense BF16 equation within 0.00439453125 and 0.001953125 respectively.
This operator contains no model projection and therefore isolates the
attention paths directly.

| Chapter | Measured checkpoint | Primary result | Change from the preceding comparable path |
|---|---|---|---|
| Day 1 | Continuous scheduler | Defines request turnover and active-batch throughput. | Establishes the serving workload. |
| Day 2 | Chunked admission with dense reconstruction | 711.18 prefill; 35.23 output; 57.59 decode tok/s | Establishes the dense serving baseline. |
| Day 3 | Paged storage with compatibility gather | 725.46 prefill; 41.64 output; 78.53 decode tok/s | +18.2% output; +36.4% decode; -50.6% copy volume. |
| Day 4 | Correct direct paged behavior | 105.01 aggregate decode tok/s in the cumulative endpoint | Removes dense K/V reconstruction; this corpus does not isolate Day 4's scalar prefill. |
| Day 5 | BF16 long-prefill tiled schedule | No isolated scalar-versus-tiled row | The cumulative serving row below includes Day 5 but is not causal evidence for it. |

Day 1 introduces scheduling, not a kernel speedup. Day 2 makes the hidden cost
measurable: appending one token still reconstructs a padded dense batch. Day 3
makes pages canonical but retains `gather_dense()` as a compatibility
checkpoint. Day 4 then removes that compatibility movement for every query
shape. Day 5 changes only the internal schedule for supported BF16 long
prefill.

Days 4 and 5 share the final direct-paged process: queries with `L <= 8`
dispatch to the Day 4 decode schedule, supported BF16 long-prefill calls use
the Day 5 tiled schedule, and generic shapes retain a direct scalar fallback.
The phase timers report decode and prefill throughput inside the same request
trace; they do not isolate the Day 5 schedule.

Every headline number above comes from the same continuous-batch campaign. The
cumulative serving endpoints are:

| Storage and attention path | Prefill tok/s | Output tok/s | Decode tok/s | Requests/s | Peak KV MiB | Avoidable KV copy MiB |
|---|---:|---:|---:|---:|---:|---:|
| Dense growth and reconstruction | 711.18 | 35.23 | 57.59 | 0.469 | 1,096 | 209,532 |
| Paged storage plus dense gather | 725.46 | 41.64 | 78.53 | 0.555 | not a total peak | 103,445 |
| Direct paged attention | 672.68 | 46.36 | 105.01 | 0.618 | 576 | 504 |

The same raw serving artifact reports synchronized decode-call latency and the
completion gaps that include intervening prefill and scheduler work:

| Path | Decode step median / p95 / max | Completion gap median / p95 / max |
|---|---:|---:|
| Dense reconstruction | 51.03 / 84.49 / 124.52 ms | 53.16 / 248.30 / 309.74 ms |
| Paged + gather | 39.80 / 52.79 / 80.09 ms | 41.82 / 225.64 / 261.38 ms |
| Direct paged | 28.97 / 36.78 / 63.04 ms | 30.16 / 222.18 / 239.49 ms |

The compatibility row omits peak storage because an exact peak must include
both the page pool and temporary dense staging allocation. Its other counters
remain directly comparable.

Direct paged attention is 5.4% lower on prefill, 31.6% higher on output/request
throughput, 82.3% higher on decode, and 47.4% lower on measured peak KV storage
relative to dense serving. Avoidable logical copy volume falls by 99.76%.
Relative to paged storage plus gather, it is 7.3% lower on prefill, 11.3%
higher on output/request throughput, 33.7% higher on decode, and removes 99.51%
of the remaining copy volume. These cumulative system results do not isolate
the Day 5 prefill kernel or prove a short-chunk FlashAttention win.

The 8K static run remains a secondary kernel diagnostic, not a Week 3 headline
or acceptance result. At that shape, the Week 3 seam plus course paged path
raises prefill from the Week 2 path's 323.96 to 463.69 tok/s, a 43.1% gain, and
reaches 72.5% of the 639.73 tok/s full-MLX row. This does not isolate the
projection seam, measure request turnover or admission capacity, or establish
long-context support. One-token decode continues to dispatch to the Day 4
vector schedule.

### Separate causal projection-seam result

Task #360 holds the Week 3 mechanisms fixed and changes only projection
ownership on measured source `170211be3503c0ec0b1fa75bbb3b0c23a86bd3ac`:

| Causal comparison | MLX seam effect versus inherited Week 2 projections |
|---|---:|
| Chunked prefill, step 512 | +10.64% prefill; +11.91% output |
| Chunked prefill, step 128 | +11.74% prefill; +11.76% output |
| Dense Day 3 | +12.17% prefill; +16.82% output; +18.86% decode |
| Serving | +7.72% prefill; +9.42% output; +13.02% decode |

Full MLX remains 17.83% faster than the dense Day 3 seam on prefill
(equivalently, the seam is 15.13% below full MLX), because the seam changes
projections only. These causal percentages explain the ownership decision;
the task #367 tables above provide current absolute values.

The checked-in final-main corpus contains the complete raw samples, exact
source commit and tracked-clean flag, host, configuration, execution order,
and—where requests are generated—the exact request trace and its checksum:

- `benchmark_results/task367-final-main/raw/week2-32-final-main.json`
- `benchmark_results/task367-final-main/raw/week2-128-final-main.json`
- `benchmark_results/task367-final-main/raw/week2-2048-final-main.json`
- `benchmark_results/task367-final-main/raw/week2-prefill-operators-final-main.json`
- `benchmark_results/task367-final-main/raw/week3-chunked-prefill-final-main.json`
- `benchmark_results/task367-final-main/raw/week3-attention-final-main.json`
- `benchmark_results/task367-final-main/raw/week3-serving-final-main.json`
- `benchmark_results/task367-final-main/raw/week3-8k-final-main.json`

Verify the manifest, all eight raw files, and the evidence ledger with:

```bash
(cd benchmark_results/task367-final-main && \
  shasum -a 256 -c task367-final-main-sha256.txt)
```

Copy counters report logical operation volume, not hardware DRAM traffic.
Dense volume includes old K/V copied during each request-cache growth and live
K/V copied into a newly padded batch tensor at every decode step. Paged volume
includes old physical pages copied only when a layer's geometric pool grows.
Appending a token writes only its page slice, and later requests reuse freed
pages.

The raw counters make reuse, fragmentation, logical copy volume, and measured
KV headroom visible; static single-request latency cannot. Logical copy volume
is not hardware DRAM traffic, and none of these counters establishes admission
capacity without a memory-capped sweep.

The workload validates continuous batching, chunked prefill, incremental
growth, and page reuse. Prefix sharing and speculative decoding require
separate traces with shared prefixes or cache rewind events and are not claimed
by this result.


## Week 2 Profiling Boundary

The current learner interface consists of the nine public checkpoints listed at
the top of this appendix. The optional evidence-reading chapter explains how to
audit preserved packages. Older runner defaults belong to archived progressions
and must not be presented as reproducible commands for this successor.

## Optimization Map

| Observed work | Current change | Chapter |
|---|---|---|
| Full-prefix recomputation | Dense request KV cache | Week 2 Day 1 |
| Repeated logical-prefix copies | Request-bounded capacity and slice writes | Week 2 Day 2 |
| Dense projection weight traffic | Packed W4 matvec | Week 2 Day 3 |
| Poor matrix-row reuse | SIMD-group matrix prefill | Week 2 Day 4 |
| RMSNorm rereads and small primitive graphs | Register-cached RMSNorm, RoPE, SwiGLU | Week 2 Day 5 |
| Materialized dense prefill scores | BQ32/BK16 online-softmax attention | Week 2 Day 6 |
| Cumulative mechanism interaction | Exact selected feature set and fresh ledger | Week 2 Day 7 |

This map optimizes one request. It does not claim batching, paging, request
scheduling, production policy, or an unsupported long-context product result.

{{#include copyright.md}}
