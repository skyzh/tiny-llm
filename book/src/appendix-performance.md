# 🚧 Appendix: Performance Evidence Ledger

> **Status: Experimental, single-machine evidence.** See the
> [Week 2 decision guidance](./week2-overview.md#verification-status) before
> treating a correctness, integration, or performance result as broader proof.

This appendix records the measurements that determined the course order. The
numbers are not additive promises: after one bottleneck shrinks, every other
operator becomes a larger fraction of model time.

## Benchmark Method

The progression runner launches every checkpoint in a fresh process,
alternates comparison order, performs complete-request warmups, synchronizes
lazy MLX work inside the timer, and reports the median. The checked Week 2
baseline generates exactly 128 tokens at five prompt lengths:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --matrix \
  --variant week2-simd-matmul --variant mlx --repeats 2 \
  --model qwen3-4b --output-len 128 --warmup 2 \
  --prefill-logits last --json-output week2-product-matrix.json

pdm run bench-serving-progression --offline --repeats 4 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 \
  --prefill-step 128 --warmup 1 --cooldown-seconds 1 \
  --json-output benchmark_results/task367-final-main/raw/week3-serving-final-main.json
```

`--prefill-logits last` is a generation-serving workload: both the reference
solution and MLX project only the last prompt row into vocabulary logits. Use
`--prefill-logits all` for prompt scoring, but never compare the two modes.
Token 1 is selected during prefill and belongs to TTFT. Decode throughput and
TPOT use the remaining 127 intervals.

MLX's published `mlx_lm.benchmark` table uses a 2,048-token prompt and 128
generated tokens. That makes 2K/128 a useful static-library comparison point,
not a paging acceptance test or a long-context proof. Use a context sweep:

| Point | Purpose |
|---:|---|
| 128 | short-request regression control |
| 512 | intermediate short prompt and `llama-bench`-style prompt length |
| 2,048 | standard MLX-style static stress comparison |
| 8,192 | long-context attention and KV-cache stress |
| 32,640 | native endpoint when followed by exactly 128 generated tokens |

`llama-bench` commonly uses prompt-processing 512 and token-generation 128 by
default, which is another reminder that benchmark lengths are conventions, not
universal workloads. Always publish the exact prompt and output lengths.

The measured machine below is an Apple M4 Pro with a 20-core GPU and 64 GB of
memory. The current Week 2 control uses two complete warmups and two fresh
processes at each prompt length; the continuous-serving rows use one warmup and
the median of four balanced fresh processes. A 32,768-token prompt is
prefill-only because it leaves no native context room for output. Treat 65K or
131K as YaRN-qualified or synthetic stress rather than native product points.

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
| Long-context dense-KV attention (optional) | 1–2 query rows, 32 query heads / 8 KV heads / D=128, BF16 I/O, FP32 online-softmax state, tails, scale, and causal/explicit masks | Avoid full score rows while walking contiguous K/V as decode context grows | Decision pending: require ≥5% lower 8K median TPOT, favorable direction in 3/4 context pairs, and ≤2% regression at 128/512 | Readable dense grouped attention is the unsupported-shape and disable-only control | Extrapolating the retired context-256 dispatch, claiming prefill or paging, or omitting the selection witness |
| SIMD-matrix prefill | W4/group-128 layout, BF16 storage, FP32 tile accumulation, and correct partial tiles | Reuse activation and dequantized-weight tiles across prompt rows | Required path for `M > 8`; partial and new model shapes need both correctness and timing sweeps | The Python `mlx.core` matmul is the correctness oracle; Day 3 matvec remains the short-row dispatch and vanilla Metal is a bring-up control | Comparing all-logit course prefill with last-logit MLX serving |
| Fused packed-W4 gate+up + SwiGLU | Gate/up share one activation; W4 group 128, BF16 I/O, FP32 accumulation, correct tails; down stays separate | Reuse shared-input work and remove the intermediate gate/up-to-SwiGLU dispatch | Decision pending: require ≥5% targeted-phase gain at 512 or 2K, favorable direction in 3/4 row pairs, and ≤2% regression at 8K/decode | Day 5's separate gate, up, and SwiGLU path is the unsupported-shape and disable-only control | Turning a normalized isolated MLP share into a product win or allowing down-projection work into the comparison |

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

This section binds the full-model workload to the component experiments. It is
a checked single-machine example, not a portability certificate.

### Full-Model Baseline

The accepted baseline used Qwen3-4B-MLX-4bit on one Apple M4 Pro with a 20-core
GPU and 64 GB unified memory. Every request used last-row prefill logits and
generated exactly 128 tokens. Token 1 belongs to TTFT; TPOT measures the 127
later decode intervals.

| Prompt tokens | Prefill | Decode | TTFT | TPOT |
|---:|---:|---:|---:|---:|
| 128 | 829.50 tok/s | 88.52 tok/s | 154 ms | 11.298 ms |
| 512 | 854.69 tok/s | 86.07 tok/s | 599 ms | 11.618 ms |
| 2,048 | 727.19 tok/s | 75.34 tok/s | 2.816 s | 13.272 ms |
| 8,192 | 624.71 tok/s | 54.39 tok/s | 13.113 s | 18.386 ms |
| 32,640 | 370.73 tok/s | 25.71 tok/s | 88.044 s | 38.888 ms |

The final row plus 128 output tokens reaches the native 32,768-token endpoint.
A prompt of 32,768 is prefill-only; 65K and 131K require an explicit YaRN or
synthetic-stress qualification.

The two-sample medians can expose a large contradiction. They cannot turn a
small difference into a portable claim. Run the public command in
[Benchmark Method](#benchmark-method) against your own implementation and
publish the exact model, lengths, warmups, order, software, and device.

### Component Attribution

The prior single “projections” category mixed QKV/output attention projections
with gate/up/down MLP projections. The replacement isolated replay reports
model responsibilities instead:

| Phase | Normalized isolated component | Prompt 128 | Prompt 32,640 |
|---|---|---:|---:|
| Prefill | QKVO + score/softmax/value attention | 29.02% | 66.24% |
| Prefill | MLP gate/up/down + SwiGLU | 63.14% | 32.74% |
| Decode | QKVO + score/softmax/value attention | 36.24% | 62.14% |
| Decode | MLP gate/up/down + SwiGLU | 43.97% | 26.10% |

Normalization, embedding, output head, and residual/framework overhead remain
separate when the replay can measure them. These shares are normalized within
isolated replays; do not add them to full-model time or describe them as
production or fleet percentages.

The crossover selects two distinct questions. Attention is the larger
long-context target. MLP work remains the larger short-prompt prefill component
and exposes shared activation work between gate and up. Neither observation is
a candidate result.

### Days 1–5: Establish the Measured Model Path

Day 1 changes the generation algorithm: prefill once, retain dense K/V, and
send only the new token through each decode step. Days 2–5 then use matched
attribution to pack W4 weights, fuse bounded pointwise work, and choose separate
short-row and matrix-shaped projection schedules. Each step must retain its
readable fallback and survive the same full-request control.

Keep the causal order, but re-profile by component and context after each
change. A kernel that shrinks one isolated operator can reveal a different
component without improving every request shape.

### Day 6: Long-Context Dense-KV Attention

The retired attention branch stopped dispatching after context 256, so its
128-token observation cannot answer the new long-context question. The
replacement selector, `long-context-attention`, covers one- and two-row GQA at
contexts 128, 512, 2K, and 8K with FP32 online-softmax state, masks, tails, a
selection counter, and a disable-only control.

The decision is pending. Retain the candidate only with at least 5% lower 8K
median TPOT, the same favorable direction in three of four context pairs, no
more than 2% regression at 128 or 512, and disappearance of the gain when the
selector is disabled. It remains dense and contiguous; it does not claim paged
KV, chunking, paged attention, or full-prefill coverage.

At the native-endpoint prompt, a naive FP32 prefill score tensor for 32 query
heads is about 127.002 GiB. That establishes the need for memory-efficient
prefill before a native-endpoint course run; it does not turn the Day 6 decode
candidate into that solution.

### Day 7: Fused Packed-W4 Gate+Up and SwiGLU

Day 7 starts from Day 5 and is independent of optional Day 6. The
`fused-gate-up` candidate reads one activation for the two group-128 packed-W4
gate/up projections, applies SwiGLU, and leaves the down projection unchanged.
Its operator sweep uses rows 1, 32, 128, 512, and 2K, with an 8K full-request
and decode safety control.

The decision is also pending. Retain the candidate only with at least 5%
targeted-phase improvement at 512 or 2K, the same favorable direction in three
of four 32/128/512/2K pairs, no more than 2% regression at 8K or decode, and
disappearance of the gain when the selector is disabled.

The earlier Split-K candidate improved one isolated 32-token projection by
about 5% but did not improve the fixed 128-token product control. That is why it
is no longer the final Week 2 mechanism.

### Optional Capture Boundary

The optional [macOS 27 profiling lab](./week2-advanced-profiling.md) shows how
to create a trace package, hash its files, reduce gpudebug output, record a
three-sentence decision, and remove the raw package after preserving compact
evidence. Missing counter or shader trees remain unavailable rather than zero.
The synchronized product and portable isolated replay are sufficient for every
required decision.

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
| Week 2 Day 5/7 path | Course-owned zero-Steel W4 kernels, loader, SIMD-matrix helper, and any retained fused gate+up candidate | Course-owned Week 2 dense cache and operators | Week 2 course implementation versus its explicitly paired full-MLX row. |
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

The synchronized product benchmark and portable operator-attribution runner
are the required evidence path. Metal capture, Xcode visualization,
`gpudebug`, and screenshots remain optional and require macOS 27. The compact
checked result records unavailable trees instead of substituting zeros or
inferring counters; learners without that toolchain can still complete every
checkpoint and reason from the portable artifact.

## Optimization Map

| Measured bottleneck | Retained change | Chapter |
|---|---|---|
| Full-prefix decode recomputation | Dense request KV cache | Week 2 Day 1 |
| Dense projection weight traffic | Packed W4A16 x4 SIMD matvec | Week 2 Day 3 |
| Repeated small graph dispatches | RMSNorm, RoPE, SwiGLU kernels | Week 2 Day 4 |
| Scalar/strided prefill projection loads | Cooperative 32×32×32 quantized matmul | Week 2 Day 5 |
| Attention share and TPOT rising with dense context | Optional long-context online-softmax decode attention with dense fallback | Week 2 Day 6 |
| Repeated shared-input packed-W4 MLP work | Fused gate+up projections and SwiGLU with Day 5 fallback | Week 2 Day 7 |
| Functional whole-cache page updates | Aliasing page-slice write primitive | Week 3 Day 3 |
| Scalar paged final reduction | Compact D=128 SIMD reduction | Week 3 Day 4 |
| Scalar contiguous-page K/V tile loads | Cooperative paged FlashAttention loads | Week 3 Day 5 |

This is the course progression: optimize one measured cost, benchmark again,
then let the evidence choose the next chapter.

{{#include copyright.md}}
