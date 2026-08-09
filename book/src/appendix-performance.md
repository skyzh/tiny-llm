# 🚧 Appendix: Performance Evidence Ledger

> **Status: Experimental, single-machine evidence.** See the
> [Week 2 verification matrix](./week2-overview.md#verification-status) before
> treating a correctness, integration, or performance result as broader proof.

This appendix records the measurements that determined the course order. The
numbers are not additive promises: after one bottleneck shrinks, every other
operator becomes a larger fraction of model time.

## Benchmark Method

The progression runner launches every checkpoint in a fresh process,
alternates their order, performs complete-request warmups, synchronizes lazy
MLX work inside the timer, and reports the median:

```bash
pdm run bench-week2-progression --offline --repeats 4 --cooldown-seconds 1 \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-128.json

pdm run bench-serving-progression --offline --repeats 4 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 \
  --prefill-step 128 --warmup 1 --cooldown-seconds 1 \
  --json-output benchmark_results/m4-pro-qwen3-4b-week3-serving-mlx-0.32.0.json
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
memory. Static Week 2 rows use two complete warmups and the median of four
balanced fresh processes; the continuous-serving rows use one warmup and the
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
| Packed quantized matvec | W4, group size 128, BF16 parameters, contiguous packed layout, and the declared transpose convention | Read packed weights once and share unpack/scale work across SIMD lanes | Retained for `M <= 8`; multi-row prefill exposes poor reuse and motivates Day 6 | The Python `mlx.core` equation is the correctness oracle; vanilla W4 is an inspectable Metal control; named earlier checkpoints preserve the dense control | Lazy execution or timing post-materialized weights can hide weight traffic |
| RMSNorm | BF16 I/O with the sum of squares accumulated in FP32 | Fuse reduction, normalization, and weight multiply into one dispatch | Retained at Qwen hidden dimensions after both operator and decode gains; unknown dimensions require remeasurement | Python `mlx.core` RMSNorm and the Day 3 checkpoint remain selectable | Adding isolated microseconds as if checkpoint gains were independent |
| RoPE | One valid offset per batch row; even rotated dimension; tail values preserved | Fuse angle generation and pair rotation without intermediate graphs | Retained for Qwen decode rows; head-count and rotated-dimension changes require remeasurement | Python `mlx.core` RoPE and the RMSNorm-only checkpoint remain selectable | Benchmarking a cached or precomputed angle path against fresh angle construction |
| SwiGLU | Gate and up tensors have identical shape and dtype | Fuse SiLU and the gate/up product into one elementwise dispatch | Retained for Qwen MLP shapes; tiny tensors and other dtypes are not a performance claim | The Python `mlx.core` SiLU-product and the RoPE checkpoint remain selectable | Accepting an operator win without a repeated complete-model gain |
| Decode attention | `Hq % Hkv == 0`, `D <= 256`, FP32 online-softmax state, and causal/explicit mask semantics | Avoid score/probability tensors and merge softmax while walking K/V | Model dispatch is `L <= 2`, `S <= 256`, and no explicit array mask; the context sweep wins 6/6 passes through 256, while the query sweep is repeat-consistent only through `L=2` | Python `mlx.core` grouped attention handles longer queries, longer contexts, and explicit array masks | Fixed implementation order, GPU performance-state drift, extrapolating beyond 256, or treating correctness at `S=1` as schedule efficiency |
| SIMD-matrix prefill | W4/group-128 layout, BF16 storage, FP32 tile accumulation, and correct partial tiles | Reuse activation and dequantized-weight tiles across prompt rows | Required path for `M > 8`; partial and new model shapes need both correctness and timing sweeps | The Python `mlx.core` matmul is the correctness oracle; Day 3 matvec remains the short-row dispatch and vanilla Metal is a bring-up control | Comparing all-logit course prefill with last-logit MLX serving |
| Split-K prefill | Partitions align to quantization groups; partial planes are disjoint; final reduction is FP32 | Add independent groups only while the ordinary result grid is under-filled | Helps short narrow Qwen projections, is neutral around the 128-token acceptance shape, and loses once the base grid is occupied | `split_k <= 1` dispatches exactly to the Day 6 unsplit kernel | Profiling independent layers can hide under-occupancy that appears in the dependency-ordered model |

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

Week 2 has one fixed acceptance shape: Qwen3-4B, a 128-token prompt, 128 timed
decode steps, last-row logits, two complete warmups, and the median of four
fresh processes. Two passes use forward checkpoint order and two use reverse
order. The output length is 129 because prefill produces the first generated
token.

Each row is cumulative. Day 2 retains the Day 1 checkpoint while it establishes
the synchronized benchmark. Day 3 then completes the packed quantized-matvec
checkpoint.

| Chapter | Cumulative checkpoint | Prefill tok/s | Decode tok/s | Output tok/s | Change selected by the preceding evidence |
|---|---|---:|---:|---:|---|
| Day 1 | Dense request KV cache | 730.43 | 24.63 | 24.01 | Stop full-prefix decode recomputation. |
| Day 2 | Benchmark baseline | 730.43 | 24.63 | 24.01 | Measure dense projection weight traffic. |
| Day 3 | Quantized matvec | 105.00 | 58.71 | 37.95 | Keep weights packed and add the x4 decode kernel. |
| Day 4 | Fused model kernels | 105.97 | 75.21 | 44.33 | Remove the newly exposed pointwise graph launches. |
| Day 5 | Bounded decode attention | 105.99 | 75.75 | 44.50 | Historical row from before the guard extended through `S=256`; do not use it as the current checkpoint delta. |
| Day 6 | SIMD-matrix prefill | 797.45 | 75.12 | 69.17 | Fix the quantized matrix path exposed by Day 3. |
| Day 7 | Split-K prefill | 792.55 | 75.41 | 69.37 | Fill the GPU only for under-occupied short projections. |
| Baseline | MLX 0.32.0 | 830.49 | 89.37 | 81.30 | External denominator. |

The checked-in progression file predates the current `L <= 2`, `S <= 256`
guard. With the current implementation, prefill has `L=128` and stays on the
Python `mlx.core` path, while timed one-token decode steps see `S=129` through `S=256`
and enter the custom path. The historical Day 4-to-Day 5 difference is not a
current end-to-end measurement; the balanced context and query sweeps below are
the checked evidence for the production guard.

### Checked Operator Attribution That Selects Each Chapter

The checked reference-solution attribution does not replace an operator with an MLX
operator. It calls the projection, attention, pointwise, and cache paths from
`tiny_llm_ref` at Qwen3-4B shapes and replays each group at the model's real
dispatch count. The projection replay preserves the transformer dependency
order so work from a later MLP cannot hide an under-filled attention
projection. Each round rotates the category order, synchronizes every category
once, and the median follows four warmups and twelve samples. This historical
evidence is checked in for readers; reproducing it is not a learner requirement.

The bar widths below are normalized within a checkpoint. The time at the right
is the sum of the synchronized category medians, not a throughput measurement.
Forcing category boundaries prevents some whole-graph fusion, so use the shares
to rank work and the fresh-process checkpoint table above to accept or reject a
change.

![Week 2 operator attribution by cumulative checkpoint](./week2-kernel-profile.svg)

This is an operator-attribution chart, not a Metal flame graph. It ranks model
operator families and explains why the course tackles the kernels in this order.

The profile makes the progression concrete:

- Cached decode spends 81.5% of attributed time in dense projections. Day 3
  therefore changes weight storage and the decode projection schedule first.
- After packed matvec, the pointwise group is 35.8% while attention is only
  4.5% at the 128-token acceptance context. Day 4 therefore removes the
  measured normalization, position, and activation overhead first.
- After the Day 4 pointwise kernels, the balanced operator sweeps isolate a
  removable attention gap through `S=256` and a repeat-consistent query-length
  win through `L=2`. Day 5 tests online softmax inside those bounds.
- At the fixed workload, 128-token prefill remains outside the query-length
  guard. Its profile makes the vanilla quantized projection path 99.0% of
  attributed prefill time, which selects the cooperative matrix kernel in Day
  6; one-token decode uses the bounded Day 5 path.
- After Day 6, projections remain most of the inherent prefill work, but the
  long-shape operator comparison is already close to MLX. The 32-token shape
  sweep then isolates under-occupied Qwen projections and selects Split-K only
  below their measured crossover.

The checked-in raw profile is
`benchmark_results/m4-pro-qwen3-4b-week2-kernel-profile-mlx-0.32.0.json`.
The balanced fresh-process samples are
`benchmark_results/m4-pro-qwen3-4b-week2-progression-mlx-0.32.0.json`.

The operator tables below use `bench-week2-operators` with twelve warmup rounds
and sixty measured rounds. Each round synchronizes every implementation, and
the runner rotates through every execution order so GPU performance-state drift
does not consistently favor Python reference code, the course kernel, or MLX. These
latencies are microbenchmarks; only the fresh-process table above accepts an
end-to-end checkpoint.

### Day 1: Cache the Prefix

The dense cache makes prefill a one-time cost, but every decode projection
still reads dense weights. Day 1 therefore starts with respectable prefill and
only 24.63 decode tok/s. The result gives Day 2 a real cached baseline to
measure.

### Day 2: Measure Before Optimizing

Day 2 changes the measurement discipline rather than the model. The end-to-end
row and synchronized attribution answer different parts of the handoff:

| Evidence | Result | Decision |
|---|---:|---|
| Complete-model decode | 24.63 tok/s; MLX 89.37 tok/s | A large decode gap remains. |
| Dense projections | 33.66 ms, 81.5% of attributed time | Optimize projection weight traffic first. |
| Pointwise operators | 6.45 ms, 15.6% | Defer until projections shrink. |
| Attention | 0.85 ms, 2.1% | Do not select attention from this workload. |
| KV growth | 0.33 ms, 0.8% | The dense cache already removed prefix recomputation. |

The operator-family result is sufficient to select the quantized-matvec work
for Day 3. The isolated packed-W4 control is not the Day 2 model's dense
projection; it remains a readable schedule comparison without pretending that
one shader ranked the complete model.

### Day 3: Keep Weights Packed

The x4 W4A16 matvec raises complete-model decode from 24.63 to 58.71 tok/s, a
138.4% gain. Prefill falls from 730.43 to 105.00 tok/s because matrix-shaped
inputs still use the vanilla Metal quantized kernel. The operator microbenchmark
checks whether the decode gain came from the intended projection schedule:

| Qwen3-4B projection, `M=1` | Vanilla Metal | Packed matvec | MLX |
|---|---:|---:|---:|
| Q | 750.3 us | 187.6 us | 183.4 us |
| K | 239.5 us | 145.1 us | 147.8 us |
| V | 244.8 us | 147.0 us | 138.9 us |
| O | 590.3 us | 163.7 us | 160.2 us |
| MLP gate | 908.8 us | 182.5 us | 177.2 us |
| MLP up | 948.0 us | 185.6 us | 182.9 us |
| MLP down | 1,243.3 us | 188.3 us | 181.6 us |
| Vocabulary head | 11,086.1 us | 1,030.2 us | 1,029.3 us |

The packed operator is close to MLX at every listed shape. Projections still
occupy 57.9% of the synchronized model replay because every layer inherently
uses them, but normalization, position, and activation now occupy 35.8% and are
the larger removable gap. That combination, rather than the absolute height of
the projection bar, selects Day 4.

### Day 4: Fused Model Kernels

The cumulative model and operator results agree on all three retained changes:

| Checkpoint | Decode tok/s | Python reference | Fused operator | MLX operator |
|---|---:|---:|---:|---:|
| Day 3 packed matvec | 58.71 | -- | -- | -- |
| Fast RMSNorm | 65.94 | 210.0 us | 168.2 us | 147.1 us |
| Fast RoPE | 71.16 | 180.9 us | 144.8 us | 118.7 us |
| Fused SwiGLU | 75.21 | 189.4 us | 125.7 us | 137.2 us |

The pointwise group falls from 35.8% after Day 3 to 10.5%. Projections are now
80.5% of attributed decode time but are already close to their MLX operator
latencies. A direct dispatch trace can verify that the RMSNorm, RoPE, and
SwiGLU pipelines all ran. The balanced
`S=32,128,160,192,256` sweep then isolates an attention opportunity through the
largest measured context; the query-length sweep supplies the other dispatch
boundary.

### Day 5: Fused Decode Attention

The matched short-context model checkpoint uses a 32-token prompt and an output
length of 97. Prefill produces the first token, so all 96 timed decode calls
grow the cache from `S=33` through `S=128` and enter the custom guard. Under
that workload, fused attention raises median decode from 59.90 to 61.78 tok/s
(+3.1%) and output throughput from 48.52 to 49.54 tok/s (+2.1%). MLX reaches
68.86 decode tok/s, so the bounded checkpoint reaches 89.7% of that matched
denominator. The raw samples are checked in at
`benchmark_results/m4-pro-qwen3-4b-week2-short-context-mlx-0.32.0.json`.

The current context sweep includes the FP32 promotion and output cast used by
the Python `mlx.core` fallback. It uses six forward/reverse context passes, rotates
every implementation order, and retains 60 samples per implementation and
pass:

| Cached context | Python reference | Fused | MLX | Fused vs Python | Pass wins |
|---:|---:|---:|---:|---:|---:|
| 32 | 143.0 us | 125.7 us | 116.3 us | 1.138x | 6/6 |
| 128 | 149.3 us | 136.3 us | 120.6 us | 1.095x | 6/6 |
| 160 | 151.2 us | 140.1 us | 120.9 us | 1.079x | 6/6 |
| 192 | 154.0 us | 143.9 us | 121.9 us | 1.071x | 6/6 |
| 256 | 158.0 us | 150.7 us | 122.8 us | 1.048x | 6/6 |

The query-length sweep holds `S=128`, Qwen3-4B's 4:1 GQA ratio, and the causal
form while balancing L1/L2/L4/L8 order over six passes:

| Query length | Python reference | Fused | MLX | Fused vs Python | Pass wins |
|---:|---:|---:|---:|---:|---:|
| 1 | 244.4 us | 213.1 us | 155.9 us | 1.147x | 6/6 |
| 2 | 341.4 us | 258.8 us | 185.3 us | 1.319x | 6/6 |
| 4 | 322.7 us | 297.3 us | 197.4 us | 1.085x | 4/6 |
| 8 | 377.7 us | 491.5 us | 290.6 us | 0.768x | 0/6 |

At `L=1`, the causal mask permits the entire existing cache and is equivalent
to unmasked one-token decode; longer rows measure causal multi-token chunks.
The context sweep supports `S <= 256`, while `L=2` is the largest
repeat-consistent query-length win. Those results define the current
`L <= 2`, `S <= 256` guard. The checked raw records are
`benchmark_results/m4-pro-qwen3-4b-week2-attention-context-sweep-mlx-0.32.0.json`
and
`benchmark_results/m4-pro-qwen3-4b-week2-attention-query-sweep-mlx-0.32.0.json`.

In the fixed 128-token workload, prefill remains outside the query-length guard
and attributes 1,196.34 ms of 1,208.78 ms, or 99.0%, to quantized projections;
attention accounts for 6.08 ms and the pointwise group for 6.35 ms. That
prefill bottleneck selects the matrix-shaped projection kernel in Day 6.

### Day 6: Use Cooperative Loads for Quantized Prefill

At the fixed-workload prefill checkpoint, quantized projections account for 1,196.34 ms
of the 1,208.78 ms attributed profile, or 99.0%. The cooperative matrix
schedule reduces attributed projection time to 147.63 ms and raises
complete-model prefill from 105.99 to 797.45 tok/s. MLX reaches 830.49 tok/s.

The long-row controls show that the tile arithmetic and cooperative loads are
healthy once the result grid is occupied:

| Projection at `M=2048` | Day 6 | MLX |
|---|---:|---:|
| Q, `2560 -> 4096` | 6.64 ms | 6.65 ms |
| K, `2560 -> 1024` | 1.78 ms | 1.79 ms |
| O, `4096 -> 2560` | 6.82 ms | 6.66 ms |
| MLP gate, `2560 -> 9728` | 15.65 ms | 15.75 ms |
| MLP down, `9728 -> 2560` | 16.25 ms | 15.90 ms |

At the 128-token acceptance shape, the same projections are within 3.1% of
MLX. The short-row control exposes a different pattern:

| Projection at `M=32` | Day 6 SIMD | MLX | Gap |
|---|---:|---:|---:|
| Q | 733.0 us | 643.9 us | 13.8% |
| K | 221.0 us | 205.1 us | 7.8% |
| O | 256.3 us | 241.4 us | 6.2% |
| MLP gate | 410.0 us | 406.6 us | 0.8% |
| MLP down | 440.1 us | 391.7 us | 12.4% |

The operator gaps correlate with result-grid size rather than reduction width
or arithmetic. For the narrow K projection, the unsplit launch geometry is:

| Prompt rows | Row tiles | Output tiles | Independent threadgroups |
|---:|---:|---:|---:|
| 32 | 1 | 32 | 32 |
| 128 | 4 | 32 | 128 |
| 2,048 | 64 | 32 | 2,048 |

The dispatch formula yields 32 independent threadgroups for the first row of
this table. The long controls rule out a generally slow schedule, while the short operator
table and calculated dispatch geometry select Split-K for Day 7.

### Day 7: Split K Only Below the Crossover

The per-projection microbenchmark tests the proposed occupancy fix directly at
`M=32`:

| Projection | Day 6 SIMD | Split-K | MLX | Split-K effect |
|---|---:|---:|---:|---:|
| Q | 733.0 us | 612.3 us | 643.9 us | 1.20x faster |
| K | 221.0 us | 201.3 us | 205.1 us | 1.10x faster |
| O | 256.3 us | 243.8 us | 241.4 us | 1.05x faster |
| MLP gate | 410.0 us | 414.2 us | 406.6 us | Falls back; within noise |
| MLP down | 440.1 us | 395.1 us | 391.7 us | 1.11x faster |

The complete 32-token model confirms that the useful projection changes survive
composition:

| Checkpoint | Prefill tok/s | Decode tok/s | Prefill / MLX |
|---|---:|---:|---:|
| Day 6 cooperative matmul | 607.36 | 83.53 | 82.3% |
| Day 7 split-K | 679.50 | 83.53 | 92.1% |
| MLX 0.32.0 | 737.55 | 90.52 | 100% |

Split-K adds 11.9% complete-model prefill at this short shape. At `M=128`, the
operator sweep is neutral: Q, O, gate, and down dispatch unchanged, while the
narrow K split measures 221.2 us versus 222.8 us unsplit. The fresh-process
acceptance result is likewise neutral at 792.55 versus 797.45 prefill tok/s. At
`M=2048`, every projection falls back exactly to Day 6. Because the dispatch
geometry, operator table, and end-to-end result agree. The direct dispatch trace
must show the accumulation and merge pipelines, while the calculated policy
supplies the partition count and the shape sweep
decides where those costs are worthwhile.

The fresh-process samples for the short control point are checked in at
`benchmark_results/m4-pro-qwen3-4b-week2-32-mlx-0.32.0.json`. The completed
Week 2 path reaches 95.4% of MLX prefill, 84.4% of MLX decode, and 85.3% of MLX
end-to-end output throughput at the 128-token acceptance shape. Both required
phase ratios exceed 80%. Longer static sweeps remain attention diagnostics;
they do not test the memory-management reasons for paging.

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
  --json-output benchmark_results/m4-pro-qwen3-4b-week3-serving-mlx-0.32.0.json
```

A complete warmup compiles the kernels. The runner then synchronizes and resets
every page pool, so the measured paged run starts with zero pages and zero
backing capacity.

The Days 1–2 chunk-size control uses one deterministic Qwen3-0.6B trace with
seed 0, eight 64–512-token prompts, a fixed 32-token output budget, and four
balanced fresh processes. A gap is measured between synchronized decode-call
completions only while a decode request is active:

| Prefill budget | Output tok/s | Requests/s | Decode step p95 | Decode gap p95 / max |
|---:|---:|---:|---:|---:|
| 32 | 105.47 | 3.296 | 17.52 ms | 30.39 / 32.47 ms |
| 128 | 144.91 | 4.528 | 18.78 ms | 46.52 / 48.80 ms |
| 512 | 157.00 | 4.906 | 19.57 ms | 76.04 / 122.16 ms |

Because 512 covers every prompt in this trace, that row is the full-prompt Day
1 control. The monotonically smaller p95 gap at smaller budgets comes with
lower throughput; 128 is the measured course compromise.

The Day 4 operator control uses `B=1`, `Hq=32`, `Hkv=8`, `L=1`, `D=128`, BF16,
and 128-token pages. Each row is the median of four balanced fresh-process
medians, each containing 60 synchronized calls after five warmups:

| Context | Dense + gather | Direct paged | MLX fused |
|---:|---:|---:|---:|
| 128 | 184.01 us | 187.55 us | 153.59 us |
| 1,024 | 420.88 us | 249.79 us | 207.18 us |

The direct operator is 1.9% slower than dense-plus-gather at 128 tokens and
40.7% faster at 1,024 tokens. MLX remains faster at both shapes. All three
paths pass the checked BF16 correctness tolerance before timing.

| Chapter | Measured checkpoint | Primary result | Change from the preceding comparable path |
|---|---|---|---|
| Day 1 | Continuous scheduler | Defines request turnover and active-batch throughput. | Establishes the serving workload. |
| Day 2 | Chunked admission with dense reconstruction | 718.30 prefill; 32.54 output; 50.42 decode tok/s | Establishes the dense serving baseline. |
| Day 3 | Paged storage with compatibility gather | 730.69 prefill; 38.44 output; 65.88 decode tok/s | +18.1% output; +30.6% decode; -50.6% copy volume. |
| Day 4 | Direct paged decode schedule | 82.11 aggregate decode tok/s | +24.6% decode over the compatibility gather path. |
| Day 5 | Complete direct paged path | 679.56 prefill; 41.88 output; 82.11 decode tok/s | +28.7% output and request throughput over dense serving. |

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
| Dense growth and reconstruction | 718.30 | 32.54 | 50.42 | 0.433 | 1,096 | 209,532 |
| Paged storage plus dense gather | 730.69 | 38.44 | 65.88 | 0.512 | — | 103,445 |
| Direct paged attention | 679.56 | 41.88 | 82.11 | 0.558 | 576 | 504 |

The same raw serving artifact reports synchronized decode-call latency and the
completion gaps that include intervening prefill and scheduler work:

| Path | Decode step median / p95 / max | Completion gap median / p95 / max |
|---|---:|---:|
| Dense reconstruction | 58.95 / 90.60 / 169.01 ms | 62.61 / 241.29 / 344.04 ms |
| Paged + gather | 48.86 / 53.64 / 57.82 ms | 50.65 / 221.90 / 233.89 ms |
| Direct paged | 38.27 / 39.83 / 43.46 ms | 39.13 / 224.70 / 240.99 ms |

The compatibility row omits peak storage because an exact peak must include
both the page pool and temporary dense staging allocation. Its other counters
remain directly comparable.

Direct paged attention improves output and request throughput by 28.7%,
aggregate decode by 62.8%, and peak KV storage by 47.4% relative to dense
serving. Avoidable logical copy volume falls by 99.8%. Relative to paged
storage plus gather, the direct operator adds 9.0% output throughput, 24.6%
decode throughput, and removes 99.5% of the remaining copy volume. Prefill is
5.4% below dense and 7.0% below gather at the 128-token serving chunk, so the
chapter does not claim a short-chunk FlashAttention speedup.

The 8K static run remains a secondary kernel diagnostic, not a Week 3 headline
or acceptance result. At that shape, paged FlashAttention raises prefill from
323.26 to 424.14 tok/s. MLX reaches 594.21 tok/s, so the complete Week 3 path in
the reference solution reaches 71.4%. This shows where the cumulative paged
prefill path begins to help without mixing a static denominator into the
serving progression.
One-token decode continues to dispatch to the Day 4 vector schedule.

The checked-in Week 3 files contain the complete raw samples, exact source
commit and tracked-clean flag, host, configuration, execution order, and—where
requests are generated—the exact request trace and its checksum:

- `benchmark_results/m4-pro-qwen3-0.6b-week3-chunked-prefill-mlx-0.32.0.json`
- `benchmark_results/m4-pro-qwen3-4b-week3-attention-mlx-0.32.0.json`
- `benchmark_results/m4-pro-qwen3-4b-week3-8k-mlx-0.32.0.json`
- `benchmark_results/m4-pro-qwen3-4b-week3-serving-mlx-0.32.0.json`

Verify all four file hashes from the repository root with:

```bash
shasum -a 256 -c benchmark_results/m4-pro-week3-evidence-mlx-0.32.0.sha256
```

Copy counters report logical operation volume, not hardware DRAM traffic.
Dense volume includes old K/V copied during each request-cache growth and live
K/V copied into a newly padded batch tensor at every decode step. Paged volume
includes old physical pages copied only when a layer's geometric pool grows.
Appending a token writes only its page slice, and later requests reuse freed
pages.

The direct-paged median reaches 1,116 live pages out of 1,152 reserved pages,
reuses 2,196 page allocations, and records 15,840 unused tail slots across
layer caches. At the same peak-tail-waste snapshot, all live pages contain
133,632 token slots, so tail waste is 11.9%, or 61.9 MiB of KV storage. This
denominator excludes unused reserved pool capacity. The run grows the layer
pools 144 times because it starts empty. These counters make reuse,
fragmentation, and measured KV headroom visible; static single-request latency
cannot. They do not establish admission capacity without a memory-capped
sweep.

The workload validates continuous batching, chunked prefill, incremental
growth, and page reuse. Prefix sharing and speculative decoding require
separate traces with shared prefixes or cache rewind events and are not claimed
by this result.

## Week 2 Profiling Boundary

The balanced JSON tables and SVG above are the checked-in evidence for the
current course. Learners are not required to generate Metal captures, Xcode
visualizations, `gpudebug` reports, profiling microbenchmarks, or screenshots.
The full profiling workflow will return when the macOS 27 tooling is available;
until then, matched synchronized benchmarks are the acceptance evidence.

## Optimization Map

| Measured bottleneck | Retained change | Chapter |
|---|---|---|
| Full-prefix decode recomputation | Dense request KV cache | Week 2 Day 1 |
| Dense projection weight traffic | Packed W4A16 x4 SIMD matvec | Week 2 Day 3 |
| Repeated small graph dispatches | RMSNorm, RoPE, SwiGLU kernels | Week 2 Day 4 |
| Growing short-context attention | Online-softmax decode kernel | Week 2 Day 5 |
| Scalar/strided prefill projection loads | Cooperative 32×32×32 quantized matmul | Week 2 Day 6 |
| Under-filled short-prefill result grid | Measured split-K dispatch | Week 2 Day 7 |
| Functional whole-cache page updates | Aliasing page-slice write primitive | Week 3 Day 3 |
| Scalar paged final reduction | Compact D=128 SIMD reduction | Week 3 Day 4 |
| Scalar contiguous-page K/V tile loads | Cooperative paged FlashAttention loads | Week 3 Day 5 |

This is the course progression: optimize one measured cost, benchmark again,
then let the evidence choose the next chapter.

{{#include copyright.md}}
