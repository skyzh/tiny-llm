# 🚧 Appendix: Performance Evidence Ledger

> **Historical evidence from an earlier full Week 2 course state.** The
> [current Week 2 route](./week2-overview.md) ships Days 1–5, ending at
> [tiled dense attention and `selected`](./week2-05-tiled-prefill-attention.md). The
> task #367's legacy Week 2 and Week 3 tables below belong to predecessor
> source `18aec8503929d80c986324578068ecac2463c2ac`; other historical runs
> state their own source. The example runner commands now write fresh results
> outside the tracked corpus; they do not reproduce those figures on this
> checkout.

This appendix records the measurements that determined the course order. The
numbers are not additive promises: after one bottleneck shrinks, every other
operator becomes a larger fraction of model time.

## Benchmark Method

The progression runner launches every checkpoint in a fresh process,
alternates their order, performs complete-request warmups, synchronizes lazy
MLX work inside the timer, and reports the median. These commands use current
runner syntax to collect new data; their output files belong to you and do
not replace the predecessor samples:

```bash
benchmark_result_root="$HOME/tiny-llm-benchmark-results"
mkdir -p "$benchmark_result_root"
benchmark_result_dir="$(mktemp -d "$benchmark_result_root/run-XXXXXX")"

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last \
  --json-output "$benchmark_result_dir/week2-128-tiny-llm.json"

pdm run bench-serving-progression --offline --repeats 4 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 \
  --prefill-step 128 --warmup 1 --cooldown-seconds 1 \
  --json-output "$benchmark_result_dir/week3-serving-ref.json"
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
memory. The current Week 2 control uses two complete warmups and two balanced
fresh processes; the continuous-serving rows use one warmup and the
median of four balanced fresh processes.

## Week 2 Checkpoint Retention Ledger

A polished explanation is not evidence that an optimization belongs in the
course. Before retaining a checkpoint, answer six questions: its invariant,
why it could be faster, where it wins, where it loses, its fallback, and how the
benchmark could mislead us. This ledger records the current answers; links
below contain the measurements.

| Checkpoint | Required invariant | Performance hypothesis | Retained range and losing shapes | Fallback or control | Main benchmark trap |
|---|---|---|---|---|---|
| Dense KV cache | Caller offset equals every layer cache length; K/V append on the sequence axis | Reuse projected prefix K/V instead of recomputing the full model prefix | Wins incremental decode as the prefix grows; repeated `concat` still copies `O(S²)` bytes | Week 1 full-prefix model remains the semantic control; Week 3 pages replace growth copies | Comparing cached MLX with an uncached course model measures different algorithms |
| Packed quantized matvec | W4, group size 128, BF16 parameters, contiguous packed layout, and the declared transpose convention | Read packed weights once and share unpack/scale work across SIMD lanes | Retained for `M <= 8`; multi-row prefill exposes poor reuse and motivates Day 5 | The Python `mlx.core` equation is the correctness oracle; vanilla W4 is an inspectable Metal control; named earlier checkpoints preserve the dense control | Lazy execution or timing post-materialized weights can hide weight traffic |
| RMSNorm | BF16 I/O with the sum of squares accumulated in FP32 | Fuse reduction, normalization, and weight multiply into one dispatch | Retained at Qwen hidden dimensions after both operator and decode gains; unknown dimensions require remeasurement | Python `mlx.core` RMSNorm and the Day 3 checkpoint remain selectable | Adding isolated microseconds as if checkpoint gains were independent |
| RoPE | One valid offset per batch row; even rotated dimension; tail values preserved | Fuse angle generation and pair rotation without intermediate graphs | Retained for Qwen decode rows; head-count and rotated-dimension changes require remeasurement | Python `mlx.core` RoPE and the RMSNorm-only checkpoint remain selectable | Benchmarking a cached or precomputed angle path against fresh angle construction |
| SwiGLU | Gate and up tensors have identical shape and dtype | Fuse SiLU and the gate/up product into one elementwise dispatch | Retained for Qwen MLP shapes; tiny tensors and other dtypes are not a performance claim | The Python `mlx.core` SiLU-product and the RoPE checkpoint remain selectable | Accepting an operator win without a repeated complete-model gain |
| Decode attention (optional lab) | `Hq % Hkv == 0`, `D <= 256`, FP32 online-softmax state, and causal/explicit mask semantics | Avoid score/probability tensors and merge softmax while walking K/V | The checked fixed-workload result is equivocal; retain only for an explicitly measured context and fallback | Python `mlx.core` grouped attention handles unsupported shapes and is the control | Prescribing the lab from chapter order, extrapolating one context, or promoting `n=2` product noise |
| SIMD-matrix prefill | W4/group-128 layout, BF16 storage, FP32 tile accumulation, and correct partial tiles | Reuse activation and dequantized-weight tiles across prompt rows | Required path for `M > 8`; partial and new model shapes need both correctness and timing sweeps | The Python `mlx.core` matmul is the correctness oracle; Day 3 matvec remains the short-row dispatch and vanilla Metal is a bring-up control | Comparing all-logit course prefill with last-logit MLX serving |
| Split-K prefill | Partitions align to quantization groups; partial planes are disjoint; final reduction is FP32 | Add independent groups only while the ordinary result grid is under-filled | Conditionally retained at the measured 32-token control; rejected at the fixed 128-token product workload | `split_k <= 1` dispatches exactly to the Day 5 unsplit kernel | Static dispatch does not prove occupancy, and a short-shape replay does not prove a fixed-workload gain |

This is a retention ledger, not a portability certificate. A new GPU, MLX
release, model shape, dtype, or workload reopens the corresponding row.

## Long-Context Budget for Week 4

Context length has separate model, memory, and latency limits. For the course
Qwen3-4B checkpoint, one token of BF16 K/V state occupies

```text
36 layers * 2 (K and V) * 8 KV heads * 128 values * 2 bytes
    = 147,456 bytes = 144 KiB per token
```

The checkpoint declares `max_position_embeddings = 65,536`, but its
`rope_scaling` field is empty. Qwen documents that Qwen3 training covers
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
the course checkpoint would be outside its configured and training ranges,
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

Run a new operator sweep without replacing the preserved historical JSON:

```bash
benchmark_result_root="$HOME/tiny-llm-benchmark-results"
mkdir -p "$benchmark_result_root"
benchmark_result_dir="$(mktemp -d "$benchmark_result_root/run-XXXXXX")"

pdm run bench-long-context-attention \
  --json-output "$benchmark_result_dir/long-context-attention.json"
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

This section is a checked example of the course's discover → optimize →
re-profile loop. It is not a portability certificate or a set of performance
thresholds.

### Bound Evidence

The run used exact source add389b747793e910f0506f5720dd0aac373d126
on one Apple M4 Pro with 20 GPU cores and 64 GB unified memory, macOS 27 build
26A428, gpudebug 1.0, Python 3.12.13, MLX 0.32.0, mlx-lm 0.31.3, and
Qwen3-4B-MLX-4bit from the local cache.

The fixed product control used a 128-token prompt, 129 output tokens,
last-row prefill logits, seed 0, two synchronized warmups, and two balanced
fresh-process samples. The attribution cases used four warmups and twelve
balanced synchronized iterations. With n=2, product medians can reject a
large contradiction; they cannot turn a sub-percent change into a portable
claim.

To apply the method on the current five-day checkout, use its live checkpoint
selectors and fresh user-owned output files. These commands produce new
measurements; they do not reproduce the historical results below:

~~~bash
benchmark_result_root="$HOME/tiny-llm-benchmark-results"
mkdir -p "$benchmark_result_root"
benchmark_result_dir="$(mktemp -d "$benchmark_result_root/run-XXXXXX")"

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-kv-cache --variant week2-quantized-matvec \
  --variant week2-swiglu --variant week2-simd-matmul \
  --variant week2-tiled-prefill --variant week2-selected --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last \
  --json-output "$benchmark_result_dir/week2-progression-tiny-llm.json"

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case kv-cache:decode:128 --case quantized-matvec:decode:128 \
  --case swiglu:decode:128 --case swiglu:prefill:128 \
  --case simd-matmul:prefill:128 --case tiled-prefill:prefill:128 \
  --case selected:prefill:128 --warmup 4 --iterations 12 \
  --json-output "$benchmark_result_dir/week2-attribution-tiny-llm.json"
~~~

The checked compact result is
benchmark_results/m4-pro-qwen3-4b-week2-gpudebug-macos27-mlx-0.32.0.json.
It records unavailable evidence explicitly. It contains no raw trace, absolute
workspace path, screenshot, token output, or portable timing claim.

### Day 1: Cache the Prefix

Day 1 changes the generation algorithm: prefill once, retain dense K/V state,
and send only the new token through each decode step. The matched Week 1 versus
kv-cache product observation measures that algorithmic change before any
kernel is replaced. A shader trace is not needed to justify the cache.

### Day 2: Discover the First Operator Category

The cached-decode attribution reported 34.527 ms of projection work, or 83.9%
of the attributed total. Two BF16 GEMV shaders accounted for 93.60% of the
available shader ranking. This selected dense projection weight traffic as the
first bounded target.

The evidence-to-next-change decision was: pack W4 weights, change only the
projection path, and repeat the identical decode workload. A failure to reduce
projection time, or a regression in matched product decode, would falsify the
hypothesis.

### Day 3: Keep Weights Packed

The packed W4 candidate reduced attributed projection time from 34.527 ms to
10.700 ms (-69.0%) and reduced total attributed time by 56.8%. On the
fixed-workload two-sample product control, decode rose from 24.38 to 58.90
tokens/s (+141.6%).

The next re-profile mattered as much as the speedup: normalization, position,
and activation work now occupied 5.948 ms, or 33.5% of attributed time. That
newly exposed category selected the fused Day 4 operators.

### Day 4: Fused Model Kernels

Fused RMSNorm, RoPE, and SwiGLU reduced the selected category from 5.948 ms to
1.251 ms (-79.0%) and total attributed time by 27.3%. The product control
improved at every cumulative substep: RMSNorm +10.7%, RoPE +8.7%, and SwiGLU
+4.9%.

After the full Day 4 checkpoint, projections again dominated decode at 10.516
ms / 81.4%, while attention was 0.837 ms / 6.5%. At 128-token prefill, the
portable attribution put projections at 1,201.306 ms / 99.1%. That prefill
result—not a predetermined chapter order—selected SIMD-matrix prefill for Day
5.

### Day 5: Restore Matrix-Shaped Prefill

The cooperative W4 SIMD-matrix schedule reduced attributed 128-token projection
time to 163.172 ms (-86.4%) and total attributed time by 85.8%. Fixed-workload
prefill rose from 106.44 to 721.60 tokens/s (+577.9%). The succeeding capture
ranked the SIMD-group W4 matrix shader at 96.86% of available shader cost.

These effects justify retaining the schedule for this source tree and workload.
They do not establish the same gain on another Apple GPU, model, prompt length,
or dependency version.

### Day 6: Keep the Secondary Operator Lab Optional

After Day 4, the checked decode-attention branch changed attributed attention
from 0.837 ms to 0.831 ms (-0.75%), while total attributed time rose 0.97%.
The separate product control showed a small decode change from 74.34 to 76.50
tokens/s (+2.91%). Those mixed signals support an inconclusive worked branch,
not a universal bottleneck or a prerequisite for Day 7.

The capture did confirm that the custom attention shader ran: it accounted for
10.27% of available shader cost while packed projections accounted for 82.83%.
That is useful mechanism evidence, but it does not make the optional branch the
next dominant optimization.

### Day 7: Split K Only Where the Shape Supports It

At 32-token prefill, the unsplit SIMD projection replay exposed an under-filled
schedule. Split-K reduced attributed projection time from 48.433 ms to 46.008
ms (-5.01%) and total attributed time by 4.87%. Static inspection found both
Split-K and reduction dispatches, but the replay produced no timeline, shader
ranking, or counter tree, so no occupancy improvement is inferred.

The fixed 128-token product control rejects a broad claim: prefill changed from
721.60 to 718.36 tokens/s (-0.45%) and decode changed by +0.14%. The checked
decision therefore conditionally retains Split-K for the measured short shape
and rejects it for the fixed 128-token product workload. Another device or
model needs a fresh crossover measurement.

### What the Capture Can and Cannot Add

Six of eight checked captures exposed complete shader/counter detail. The
pre-SIMD 128-token prefill capture exposed timeline counters but no shader or
command ranking. The 32-token Split-K capture exposed only static dispatch.
Missing trees remain unavailable; they are not recorded as zero and do not
support inferred counters.

The optional [macOS 27 profiling lab](./week2-advanced-profiling.md) shows how
to create a trace package, hash its files, reduce gpudebug output, record a
three-sentence decision, and remove the raw package after preserving compact
evidence. The portable benchmark and attribution path remains sufficient for
every required checkpoint.

## Week 3 Performance by Chapter

Paging adds indirect K/V reads and is not expected to beat contiguous
attention for one preallocated static request. Week 3 therefore measures a
serving workload with request turnover, incremental unknown-size growth,
chunked admission, dense batch reconstruction, and page reuse. This command
collects a new result rather than replaying the predecessor table:

```bash
benchmark_result_root="$HOME/tiny-llm-benchmark-results"
mkdir -p "$benchmark_result_root"
benchmark_result_dir="$(mktemp -d "$benchmark_result_root/run-XXXXXX")"

pdm run bench-serving-progression --offline --repeats 4 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 \
  --prefill-step 128 --warmup 1 --cooldown-seconds 1 \
  --json-output "$benchmark_result_dir/week3-serving-ref.json"
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
ablation; the latter is representative absolute evidence for predecessor
`18aec850`, not today's five-day Week 2 baseline. Do not
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

The predecessor's 8K static run remains a secondary kernel diagnostic, not a
current Week 3 headline or acceptance result. At that shape, its Week 3 seam
plus course paged path raised prefill from the former Week 2 Split-K path's
323.96 to 463.69 tok/s, a 43.1% gain, and reached 72.5% of the 639.73 tok/s
full-MLX row. The old Week 2 denominator is not today's five-day `selected`
checkpoint; a current percentage needs a fresh matched run. This does not
isolate the projection seam, measure request turnover or admission capacity,
or establish long-context support. One-token decode continues to dispatch to
the Day 4 vector schedule.

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
the task #367 tables above provide absolute values for the predecessor source,
not current measurements.

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

The synchronized product benchmark and portable operator-attribution runner
are the required evidence path. Metal capture, Xcode visualization,
`gpudebug`, and screenshots remain optional and require macOS 27. The compact
checked result records unavailable trees instead of substituting zeros or
inferring counters; learners without that toolchain can still complete every
checkpoint and reason from the portable artifact.

## Historical Optimization Map

This map records the predecessor course's seven-day Week 2 sequence. Its Day
5–7 labels do not name the [active five-day route](./week2-overview.md).

| Measured bottleneck | Retained change | Chapter |
|---|---|---|
| Full-prefix decode recomputation | Dense request KV cache | Week 2 Day 1 |
| Dense projection weight traffic | Packed W4A16 x4 SIMD matvec | Week 2 Day 3 |
| Repeated small graph dispatches | RMSNorm, RoPE, SwiGLU kernels | Week 2 Day 4 |
| Scalar/strided prefill projection loads | Cooperative 32×32×32 quantized matmul | Week 2 Day 5 |
| Explicit secondary workload | Optional online-softmax decode lab or equivalent bounded experiment | Week 2 Day 6 |
| Under-filled short-prefill result grid | Conditional measured split-K dispatch with Day 5 fallback | Week 2 Day 7 |
| Functional whole-cache page updates | Aliasing page-slice write primitive | Week 3 Day 3 |
| Scalar paged final reduction | Compact D=128 SIMD reduction | Week 3 Day 4 |
| Scalar contiguous-page K/V tile loads | Cooperative paged FlashAttention loads | Week 3 Day 5 |

The transferable method is to optimize one measured cost, benchmark again,
then let new evidence choose the next change.

{{#include copyright.md}}
