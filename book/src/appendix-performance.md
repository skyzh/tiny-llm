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
  --prefill-logits last \
  --json-output benchmark_results/task367-final-main/raw/week2-128-final-main.json

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
| Day 1 | Dense request KV cache | 706.65 | 21.73 | 21.25 | Stop full-prefix decode recomputation. |
| Day 2 | Benchmark baseline | 706.65 | 21.73 | 21.25 | Measure dense projection weight traffic. |
| Day 3 | Quantized matvec | 104.82 | 55.96 | 36.77 | Keep weights packed and add the x4 decode kernel. |
| Day 4a | Fast RMSNorm | 104.88 | 63.70 | 39.93 | Remove the first exposed pointwise graph launches. |
| Day 4b | + Fast RoPE | 105.37 | 66.20 | 40.97 | Fuse position rotation after RMSNorm. |
| Day 4c | + Fused SwiGLU | 105.84 | 67.83 | 41.65 | Fuse the remaining measured pointwise gap. |
| Day 5 | Bounded decode attention | 105.90 | 71.18 | 42.89 | Use online softmax only inside the measured guard. |
| Day 6 | SIMD-matrix prefill | 706.50 | 66.28 | 61.05 | Fix the quantized matrix path exposed by Day 3. |
| Day 7 | Split-K prefill | 707.41 | 65.83 | 60.67 | Fill the GPU only for under-occupied short projections. |
| Baseline | Full MLX 0.32.0 | 802.50 | 75.68 | 69.75 | External denominator. |

This final-main ladder exercises the current `L <= 2`, `S <= 256` decode
attention guard. The Day 5 row is therefore a current cumulative checkpoint,
not a transferred historical value. Every median recomputes from the raw
samples in `benchmark_results/task367-final-main/raw/week2-128-final-main.json`.

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
- After Day 6, projections remain most of the inherent prefill work. The
  balanced 32-token sweep isolates under-occupied Qwen projections; the
  128- and 2,048-row controls show that Split-K becomes neutral once the
  ordinary result grid is occupied. The remaining 7–11% long-row operator gap
  belongs to the base tile, not to a larger partition grid.

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
only 21.73 decode tok/s. The result gives Day 2 a real cached baseline to
measure.

### Day 2: Measure Before Optimizing

Day 2 changes the measurement discipline rather than the model. The end-to-end
row and synchronized attribution answer different parts of the handoff:

| Evidence | Result | Decision |
|---|---:|---|
| Complete-model decode | 21.73 tok/s; full MLX 75.68 tok/s | A large decode gap remains. |
| Dense projections | 33.66 ms, 81.5% of attributed time | Optimize projection weight traffic first. |
| Pointwise operators | 6.45 ms, 15.6% | Defer until projections shrink. |
| Attention | 0.85 ms, 2.1% | Do not select attention from this workload. |
| KV growth | 0.33 ms, 0.8% | The dense cache already removed prefix recomputation. |

The operator-family result is sufficient to select the quantized-matvec work
for Day 3. The isolated packed-W4 control is not the Day 2 model's dense
projection; it remains a readable schedule comparison without pretending that
one shader ranked the complete model.

### Day 3: Keep Weights Packed

The x4 W4A16 matvec raises complete-model decode from 21.73 to 55.96 tok/s, a
157.5% gain. Prefill falls from 706.65 to 104.82 tok/s because matrix-shaped
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
| Day 3 packed matvec | 55.96 | -- | -- | -- |
| Fast RMSNorm | 63.70 | 210.0 us | 168.2 us | 147.1 us |
| Fast RoPE | 66.20 | 180.9 us | 144.8 us | 118.7 us |
| Fused SwiGLU | 67.83 | 189.4 us | 125.7 us | 137.2 us |

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

At the fixed 128/129 acceptance workload, the current cumulative Day 5 row
raises decode from 67.83 to 71.18 tok/s and output throughput from 41.65 to
42.89 tok/s. Full MLX reaches 75.68 decode tok/s, so this checkpoint reaches
94.1% of that matched denominator. The short-context experiment above remains
the causal guard evidence; the final-main ladder is the representative
absolute checkpoint.

In the fixed 128-token workload, prefill remains outside the query-length guard
and attributes 1,196.34 ms of 1,208.78 ms, or 99.0%, to quantized projections;
attention accounts for 6.08 ms and the pointwise group for 6.35 ms. That
prefill bottleneck selects the matrix-shaped projection kernel in Day 6.

### Day 6: Use Cooperative Loads for Quantized Prefill

At the fixed-workload prefill checkpoint, quantized projections account for 1,196.34 ms
of the 1,208.78 ms attributed profile, or 99.0%. The cooperative matrix
schedule replaces the vanilla multi-row path and raises complete-model prefill
from 105.90 to 706.50 tok/s. Full MLX reaches 802.50 tok/s. The required
solution owns `CooperativeTileLoader` and `CooperativeBlockMMA` directly over
Metal `simdgroup_matrix`; it does not import Steel.

The long-row control shows that Split-K has no remaining occupancy problem to
solve once the result grid is full. It does not show parity with MLX:

| Projection at `M=2,048` | Day 6 SIMD | Full MLX |
|---|---:|---:|
| Q | 7,329.5 us | 6,872.2 us |
| K | 2,060.1 us | 1,902.7 us |
| V | 2,059.7 us | 1,903.0 us |
| O | 7,634.7 us | 6,906.9 us |
| MLP gate | 18,038.4 us | 16,889.7 us |
| MLP up | 18,593.4 us | 16,894.9 us |
| MLP down | 19,384.4 us | 17,421.1 us |

The SIMD latency is roughly 7–11% above MLX at the major long-row shapes. At
the 128-token acceptance shape it is roughly 5–10% above MLX, while the short
row exposes an under-filled grid:

| Projection at `M=32` | Day 6 SIMD | Split-K | Full MLX |
|---|---:|---:|---:|
| Q | 566.1 us | 513.1 us | 506.0 us |
| K | 270.9 us | 258.1 us | 235.7 us |
| V | 243.1 us | 191.4 us | 191.7 us |
| O | 287.5 us | 275.7 us | 261.3 us |
| MLP gate | 443.8 us | 448.2 us | 417.5 us |
| MLP up | 446.3 us | 443.0 us | 416.0 us |
| MLP down | 493.8 us | 448.5 us | 417.9 us |

The operator gaps correlate with result-grid size rather than reduction width
or arithmetic. For the narrow K projection, the unsplit launch geometry is:

| Prompt rows | Row tiles | Output tiles | Independent threadgroups |
|---:|---:|---:|---:|
| 32 | 1 | 32 | 32 |
| 128 | 4 | 32 | 128 |
| 2,048 | 64 | 32 | 2,048 |

The dispatch formula yields 32 independent threadgroups for the first row of
this table. The long control rejects extra reduction partitions at an occupied
grid; it does not erase the base-tile gap. The short table and calculated
geometry select a bounded Split-K experiment for Day 7.

### Day 7: Split K Only Below the Crossover

The two balanced context positions are the causal guard at `M=32`. Split-K
improves K by 29.2%/14.9%, V by 23.6%/11.3%, O by 3.9%/4.9%, and down by
11.1%/8.8%. Gate/up are neutral, and Q reverses direction (-4.5%, +1.5%), so
the pooled Q median is not a categorical win.

The complete 32-token model confirms that the useful projection changes survive
composition:

| Checkpoint | Prefill tok/s | Decode tok/s | Prefill / MLX |
|---|---:|---:|---:|
| Day 6 cooperative matmul | 537.92 | 67.97 | 76.6% |
| Day 7 split-K | 599.81 | 67.62 | 85.4% |
| Full MLX 0.32.0 | 702.61 | 77.11 | 100% |

Split-K adds 11.5% complete-model prefill at this short shape. At `M=128`, the
operator changes are small or mixed and the fresh-process result is neutral:
706.50 versus 707.41 prefill tok/s. At `M=2,048`, every projection uses the
unsplit policy and complete-model prefill is 551.48 versus 547.73 tok/s. The
direct dispatch trace must show the accumulation and merge pipelines, while
the calculated policy supplies the partition count and the shape sweep decides
where those costs are worthwhile.

The completed Week 2 path reaches 88.2% of full-MLX prefill, 87.0% of full-MLX
decode, and 87.0% of full-MLX output throughput at the fixed 128/129 acceptance
shape. Both required phase ratios exceed 80% there. The same claim is not made
at 2K or 8K, on another model, or on another GPU. Exact raw samples, process
order, and drift controls are in
`benchmark_results/task367-final-main/task367-final-main-benchmark-ledger.md`.

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
| Day 4 | Direct paged decode schedule | 105.01 aggregate decode tok/s | +33.7% decode over the compatibility gather path. |
| Day 5 | Complete direct paged path | 672.68 prefill; 46.36 output; 105.01 decode tok/s | +31.6% output/request throughput over dense serving. |

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
