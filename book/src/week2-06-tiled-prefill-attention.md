# 🚧 Week 2 Day 6: Tiled Attention

After [Day 5](./week2-05-fused-model-kernels.md), your `swiglu` checkpoint runs
projections, RMSNorm, RoPE, and SwiGLU on kernels you wrote. One large piece is
still the readable Week 1 code: attention.

Look at what readable attention does with a prompt. It multiplies every query
row by every key to build a full score matrix, applies the mask, runs softmax
over each row, and multiplies by the values. For a prompt of `L` tokens over
`S` cached positions, that score matrix has `L × S` entries for every query
head. With Qwen3-4B's 32 query heads, a 2,048-token prompt produces about 134
million scores per layer, each written to memory and read back before any
output exists.

Today you write attention that never builds that matrix. It walks over the
keys in small tiles and keeps a running summary for each query row, an
approach usually called FlashAttention. The kernel runs for prompts of nine
or more tokens. One-token decode keeps the readable path, which is already
cheap at that size.

That is the starting hypothesis for today: in your `swiglu` prefill replay
from Day 5, the attention group should have grown into a large share, and it
should grow faster than the others as prompts get longer, because the score
matrix grows with `L × S`. If your replay shows that, tiled attention should
shrink the attention group without changing the result. If attention is still
small at 128 tokens on your machine, write that down; the kernel still
matters for longer prompts, and the comparison at the end will show what it
did at this length.

Two checkpoints finish Week 2:

- **`tiled-prefill`** adds your tiled attention to the `swiglu` model;
- **`selected`** is the name of the finished Week 2 model. It turns on the same
  features as `tiled-prefill` and adds no new kernel.

## Save a Control First

Keep the cached 0.6B model and the same request for the whole day. Save the
`swiglu` result before you touch attention. The progression runner uses fresh
processes and alternates the order of variants. `--offline` needs the model
cached already:

```bash
pdm run build-ext
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-swiglu --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day6-control.json
```

The `mlx` row is a separate complete model, there for scale. Build the
reference extension and run its Day 6 tests to see the finished behavior;
your own extension still has its TODOs:

```bash
pdm run build-ext-ref
pdm run test-refsol --week 2 --day 6
```

## Softmax One Tile at a Time

The trick that makes tiling possible is that softmax can be computed in
pieces. Take one query row whose scores against four keys are
`[1, 3, 2, 0]`, with values `[10, 20, 30, 40]` (one number per key, to keep it
small). The answer is the softmax-weighted average of the values: 22.1386.

Now pretend the kernel sees the keys two at a time. For each tile it keeps
three numbers: the largest score so far `m`, the sum of `exp(score - m)`
called `l`, and the sum of `exp(score - m) × value` called `a`:

```plain
tile 1, scores [1, 3]:  m = 3,  l = e^-2 + e^0 = 1.1353,  a = 10e^-2 + 20 = 21.3534
tile 2, scores [2, 0]:  m = 2,  l = e^0 + e^-2 = 1.1353,  a = 30 + 40e^-2 = 35.4134
```

The two tiles used different reference maxima, so before adding them, rescale
each to the larger one, `m' = 3`:

```plain
l' = e^(3-3) × 1.1353 + e^(2-3) × 1.1353 = 1.5530
a' = e^(3-3) × 21.3534 + e^(2-3) × 35.4134 = 34.3812
output = a' / l' = 22.1386
```

Same answer, and at no point did the whole row of scores exist at once. In
general, merging a running state `(m, l, a)` with a new tile's `(m_b, l_b, a_b)`
is:

$$
m' = \max(m,m_b), \qquad
l' = e^{m-m'}l + e^{m_b-m'}l_b,
$$

$$
a' = e^{m-m'}a + e^{m_b-m'}a_b,
\qquad \mathrm{output} = a'/l'.
$$

Subtracting the maximum keeps every exponent at or below zero, so nothing
overflows. The one thing that is easy to get wrong: when a later tile raises
the maximum, you must rescale **both** the old sum `l` and the old accumulator
`a`.

## Task 1: Make Tiled Attention Match the Readable Result

The supplied `dense_prefill_attention_mma` adapter in
`src/tiny_llm/week2_kernels.py` prepares the arrays and calls the private
`tiny_llm_ext::_dense_attention_prefill_mma` binding, which is yours to write.
Implement that binding and `Week2DensePrefillMMA::eval_gpu` in
`src/extensions/src/week2_kernels.cpp`, plus
`week2_dense_prefill_mma_bf16_d128` in
`src/extensions/src/week2_kernels.metal`. Keep the supplied Python adapter and
header interface. Replace the `Week2DensePrefillMMA::eval_cpu` starter TODO in
`src/extensions/src/week2_kernels.cpp` with a GPU-only error. The adapter
checks shapes, makes the inputs and optional mask contiguous, calls your
primitive, and reshapes the result for the model.

The operator handles grouped-query attention in this layout:

```text
Q:      B, Hq,  L, 128
K, V:   B, Hkv, S, 128
output: B, Hq,  L, 128
```

`L` is the number of new prompt tokens and `S` includes any cached tokens
before them. Require BF16 Q, K, and V, equal K and V shapes, matching batch
sizes, and `Hq % Hkv == 0`. Query head `h` reads KV head
`floor(h / (Hq / Hkv))`; do not copy K and V out to `Hq` heads, which would
spend exactly the memory traffic grouped attention exists to save. The
operator itself rejects queries shorter than nine tokens; the model sends
those to the readable path instead.

### The Tile Schedule

Each threadgroup handles **32 query rows** (BQ32) and walks the source **16
positions at a time** (BK16), using four 32-lane SIMD groups, 128 threads in
all. First, load the threadgroup's 32 × 128 Q tile into threadgroup memory
once; it stays there for the whole walk. Then, for each source tile:

1. load a 16 × 128 K tile into threadgroup memory;
2. compute the 32 × 16 block of scores with SIMD-group matrix fragments;
3. reuse the same threadgroup storage to load the matching 16 × 128 V tile;
4. update each query row's running `m`, `l`, and `a` in FP32, as above.

After the last source tile, divide `a` by `l` and write BF16 once. In the
reference layout, the Q and K/V tiles take 12,928 bytes of threadgroup memory,
including two padding elements per row; those sizes describe the layout, not
its speed.

Guard both edges. Score cells for padding positions must add nothing to `l`
or `a`, and no load or store may run past the real rows or source positions.
The first test makes that unavoidable: it uses 33 query rows and 47 source
positions, so both the last query tile and the last source tile are partial:

```bash
pdm run build-ext
pdm run test --week 2 --day 6 -- -k task_1
```

It compares your BF16 output with MLX's causal grouped attention. Passing it
says the numbers are right, not that the kernel is fast.

This kernel never allocates an `L × S` score buffer, but it still reads all of
the dense K and V, and an additive mask passed in by a caller can itself be
`L × S`.

## Task 2: Preserve Masks and the Readable Fallback

Accept no mask, the string `"causal"`, or an additive mask array that
broadcasts over batch and query heads. Apply the mask to the scores before the
running-softmax update.

**Causal masks line up with the end of the cache.** Query row `i` may see
source positions up to `S - L + i`. With `L = 3` new tokens and `S = 7`, the
first new token sits at position 4 and can see positions 0 to 4. Drawing the
triangle as if the source were only three long would hide cached context the
model needs.

**A fully masked row must come out as zeros.** If every score in a row is
`-inf`, there is no probability mass: skip tiles with no valid score, and
write a finite zero row instead of dividing by zero and producing `NaN`. The
test for this uses nine BF16 query rows, 17 source positions, and a mask that
is `-inf` everywhere:

```bash
pdm run build-ext
pdm run test --week 2 --day 6 -- -k task_2
```

**Keep the readable path.** In `Qwen3MultiHeadAttention.__call__`, use the
readable grouped attention for one-token decode, for `L <= 8`, and for any
shape outside BF16 with head size 128. It must receive the same cache update,
scale, mask, head mapping, and output dtype as the tiled path. The reference
keeps three diagnostic counters: `tiled_prefill` (the tiled kernel ran),
`tiled_prefill_fallback` (the feature was on, but this call did not qualify),
and `readable` (the readable code ran). They help when you check routing. The
supplied tests compare outputs, not counters, so a passing long-prompt test
does not tell you decode is still correct.

If you need to move on without this kernel, keep the model and cache
interface and use the matching MLX attention operator, or the readable
grouped equation, at this one boundary. `--solution mlx` instead runs a
separate complete model.

## Task 3: Run `tiled-prefill` in the Model

Route eligible attention calls through your kernel inside `Qwen3ModelWeek2`.
The `tiled-prefill` checkpoint keeps Day 1's cache, the packed projections,
and all three Day 5 kernels; only supported prefill attention changes:

```bash
pdm run build-ext
pdm run test --week 2 --day 6 -- -k task_3
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint tiled-prefill --model qwen3-0.6b --max-tokens 16
```

This test runs `tiled-prefill` and `swiglu` on a tiny model with a 3-token
input, which must take the readable path, and a 10-token input, which
qualifies for the tiled kernel, and compares the outputs. Matching outputs do
not show which kernel ran. The 128-token benchmark below uses the tiled path
in the real model.

## Finish the Cumulative `selected` Model

`selected` has the same features as `tiled-prefill`: the chunked KV cache,
packed W4 projections, the decode matvec, SIMD matrix prefill,
register-cached RMSNorm, RoPE, SwiGLU, and tiled prefill attention, with the
short-query and wide-row fallbacks. Two of these can be switched off one at a
time through the constructor (`use_register_cached_rms_norm` and
`use_tiled_prefill_attention`), so you can measure what each one is worth. A
third switch, `use_bounded_kv_capacity`, is off by default; turning it on
gives each request's cache a hard bound of prompt plus `max_tokens`. `selected`
is also the model's default checkpoint.

```bash
pdm run test --week 2 --day 6 -- -k selected
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint selected --model qwen3-0.6b --max-tokens 16
```

The test checks that `selected` is the default, and that switching off each of
the three controls still gives close outputs. Then run the whole day:

```bash
pdm run test --week 2 --day 6
```

Even though `selected` is the default, name it explicitly in measurements so
each row says what it is.

The supplied **reference** solution adds extra scheduling to `selected` only:
native online decode, fused Q/K normalization with RoPE for four heads at its
supported shape, and submitting the first 18 layers of the 36-layer decode
graph to the GPU early, before the rest is built. These are reference optimizations, not tasks for you. The reference
`tiled-prefill` checkpoint keeps readable one-token decode, and unsupported
shapes keep their fallbacks. Run the finished reference with:

```bash
pdm run main-week2 --solution tiny_llm_ref --model qwen3-0.6b --max-tokens 16
```

This runs the reference, not your code, and your solution does not need to
reproduce those internals.

<a id="measure-the-matched-product"></a>

## Measure the Change

First compare a new `swiglu` row with the one you saved before editing. Then
compare `swiglu`, `tiled-prefill`, `selected`, and MLX with the same cached
0.6B request, machine, logit mode, and warmups:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-swiglu --variant week2-tiled-prefill \
  --variant week2-selected --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day6-product.json
```

A 128-token prompt is above the nine-token threshold, so prefill uses your
kernel. Report prefill and whole-request results separately. With
`--solution tiny_llm`, `tiled-prefill` and `selected` turn on the same
features, so do not count a difference between them as a second
improvement. With
`--solution tiny_llm_ref`, `selected` also includes the reference scheduling
above; report those rows separately and do not credit the difference to tiled
attention.

To see which kind of work dominates prefill now, replay the same step for the
three checkpoints:

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case swiglu:prefill:128 --case tiled-prefill:prefill:128 \
  --case selected:prefill:128 --warmup 4 --iterations 12 \
  --json-output week2-day6-attribution.json
```

Each case replays one 128-token prefill through the public operators that
checkpoint selects, with the same cache bound and logit setting (see
[Day 1](./week2-01-kv-cache.md) for what the replay does and does not
include). Your tiled attention is replayed in the attention group of the
`tiled-prefill` case. For your solution, `selected` selects the same
operators as `tiled-prefill`, so there is no further operator change to
credit between those two cases. If their timings still differ, keep the gap
in your notes and look into it (rerun both, check that the machine was idle,
compare the per-group medians) rather than calling it an improvement or
assuming it is noise.

The replay times operators on prepared inputs. It does not show that your
model sends 128-token prompts to the tiled kernel; check that by reading the
attention dispatch you wrote in Task 2 and Task 3. It also does not replace
the whole-request comparison above.

If you profile `--solution tiny_llm_ref` with `selected`, the profiler
reports that case as unsupported (`coverage.complete` is false, with a
reason) and gives no total or shares. That is because the reference adds
scheduling to `selected` that a public-operator replay cannot represent. It
is a missing measurement, not a zero; use `tiled-prefill` if you want a
reference replay to compare with.

An operator result, or an older 4B number, cannot show a speedup for your
0.6B checkout. Results from earlier versions of the profiler are not
comparable with this replay. The
[performance appendix](./appendix-performance.md) keeps earlier measurements,
including rows that could not be run, as history. A 4B comparison is
optional: cache the model and rerun every row at 4B.

Look at the `selected` replay one last time and write down which group leads
it now. That is the open question you carry into Week 3. The case study at
the end of this chapter shows a later build where prefill was led by the
MLP's matrix multiply.

## Where Week 2 Ends Up

How close does a finished Week 2 model get to MLX? The supplied reference has
all of Week 2's kernels in place plus the extra scheduling described above. The table
compares it with stock MLX running the same Qwen3-4B model. A value above 100% means the course model was faster.

| Prompt / generated tokens | Total throughput vs. MLX | Worst paired run | Paired runs below 95% |
|---|---:|---:|---:|
| 128 / 128 | 95.1% | 91.6% | 5 of 8 |
| 512 / 128 | 98.5% | 95.0% | 0 of 8 |
| 2,048 / 128 | 97.9% | 97.3% | 0 of 8 |
| 8,192 / 128 | 96.2% | 90.4% | 2 of 8 |
| 2,048 / 16 | 97.9% | 95.5% | 0 of 8 |
| 2,048 / 512 | 96.8% | 95.6% | 0 of 8 |
| 32,640 / 128 | 97.7% | 96.6% | 0 of 8 |
| 8,192 / 8,192 | 101.3% | 97.3% | 0 of 8 |

*Where these numbers come from:* separate measurement runs on one
Apple-silicon Mac, made on an earlier development revision of the reference
before it was merged in [PR #328](https://github.com/skyzh/tiny-llm/pull/328).
They are not a fresh timing of the merged code. Each case ran eight times on
each model, with each course-model run paired with one MLX run. The
throughput column is the ratio of the two medians (MLX's median total time
divided by the course model's), not a median of the paired ratios. All runs
used BF16 activations, W4 weights with group size 128, one request at a time,
and the same fixed number of generated tokens. PR #328 records the
remaining limits:

- in the 128 / 128 and 8,192 / 128 rows, several paired runs fell below 95%
  even though the medians did not;
- one long measurement campaign was interrupted twice, and the runs on both
  sides of each interruption are kept in the results;
- the course model's outputs differ numerically from MLX's;
- the course model failed a biology-answer check, and MLX's answer to a
  factual check had its own problem;
- runs keep generating past a natural end-of-sequence token to reach the fixed
  count, so they say nothing about output quality over the full length.

These are whole-request times, including the CPU work of preparing and
submitting GPU commands, not isolated kernel timings. The course model was
slower than MLX to produce the first token in every case. Decode speed was
mixed: faster at 2,048 / 16 and 8,192 / 8,192, slower in the rest. Its peak
allocated memory was higher than MLX's in every case except 128 / 128, by up
to about 0.9 GB at 32,640 / 128. So the result is close to MLX, not uniformly
faster. Your own runs will vary with your machine and model.

## Case Study: Profiling a Later Build

What does profiling look like further along? The two case studies below show
the same views on a much later build.

### Reading a Whole Request

The numbers in the two case studies below come from the course team's own
profiling on 6 November 2026, on one Apple silicon Mac. They measured a later
build than yours: the Week 3 serving path with Qwen3-4B, W4 weights in groups
of 128, paged attention and some experimental kernels that are not part of
the course. Use them to practise reading each view, not as targets for your
checkout. Everything you need is in the tables below; the raw captures are
not part of the course.

| Run | Wall | Prefill | Decode | Main-thread CPU | Peak MLX allocation | Of which KV |
|---|---|---|---|---|---|---|
| Short chat: 25 prompt tokens, 16 generated | 0.24 s | 0.045 s | 0.19 s | 0.07 s | 2.38 GB | 0.08 GB |
| Mixed queue: 8 requests, 1 at a time | 11.8 s | 10.2 s | 1.6 s | 0.61 s | 3.37 GB | 0.60 GB |
| Mixed queue: 8 requests, up to 4 at a time | 12.2 s | 10.9 s | 1.3 s | 0.53 s | 3.91 GB | 1.21 GB |

| Capture | Kernel | Dispatches | Share of GPU shader cost |
|---|---|---|---|
| Short chat, first decode step, whole model | quantized matrix-vector (`affine_qmv_fast…`) | 253 | 90.5% |
| | paged decode attention | 36 | 3.0% |
| | RMSNorm | 145 | 1.8% |
| First layer, 256-token prefill | MLP W4 matrix multiply (`mlp_qmm_m64_bf16`) | 3 | 65.4% |
| | other W4 matrix multiply (`affine_qmm_t…`) | 2 | 23.1% |
| | split-K W4 matrix multiply | 2 | 6.2% |
| | paged prefill attention | 1 | 2.6% |

Decode and prefill rows are separate captures; their shares do not add up
across captures. The capture replays add their own overhead, so they are not
ordinary latency either.

**Request clock first.** The short chat took 0.24 s, most of it decode. The
mixed queue of eight longer requests took about 12 s, and 86–89% of it was
prefill. The same
server has two different problems depending on what you ask it to do. For
the short chat, decode is the target; for the long queue, prefill is.

**Then the timeline.** On the short chat, the GPU was active for about 80% of
the wall time; for the remaining fifth it had nothing to run. Main-thread CPU
time was 0.07 s. CPU and GPU time overlap, so they cannot be added to get the
wall time.

**Then the kernels.** In the short chat's first decode step, a single
quantized matrix-vector kernel accounted for about 90% of GPU shader cost. In
one layer's 256-token prefill, the MLP matrix kernel accounted for 65%. The
two phases point at two different kernels, which is exactly why you profile
the phase the request clock told you to.

**Memory last.** Allowing four requests at once cut decode time from 1.6 s
to 1.3 s, but prefill grew, the whole queue took 0.4 s longer, the peak rose
from 3.37 GB to 3.91 GB and KV storage doubled to 1.21 GB. Keep results like
that one: the change helped the phase it targeted and cost the request. That is
what MLX had allocated at its peak, not a measure of physical memory in use.
When you trade memory for speed, this is the number that tells you what you
spent.

### Inside One Kernel

The second case study goes one level deeper, into the prefill MLP kernel
and a paged-attention kernel, captured once each at one representative shape
(a 256-row prefill chunk). It shows what the source-line view from
[Day 1](./week2-01-kv-cache.md#look-inside-the-gpu) adds, and also how easily it
is misread.

The MLP capture is the gate projection of layer 0, with its real weights: 256
rows of activations, 2,560 inputs, 9,728 outputs. The attention capture is
one request with 32 query heads, 8 KV heads, head size 128, 256 new rows
over a 1,280-token context, in pages of 128.

| Kernel | Source line | Weighted cost in that kernel |
|---|---|---|
| MLP | kernel entry (wrapper, inclusive) | 100.0% |
| MLP | `mma_op.mma(Xs, Ws)`: SIMD tile multiply (inclusive) | 82.1% |
| MLP | weight load and dequantize helper | 12.0% |
| Attention | inlined `simdgroup_multiply_accumulate` (inclusive of both lines below) | 75.9% |
| Attention | `Q·Kᵀ` score multiply | 38.0% |
| Attention | `P·V` output multiply | 37.9% |

| Counter (busy windows only) | MLP | Attention |
|---|---|---|
| Floating-point limiter | 96.0% | 90.5% |
| Floating-point utilization | 90.2% | 82.7% |
| L1 limiter | 20.2% | 18.1% |
| Last-level cache limiter | 18.5% | 6.1% |

**Two different "percentages".** GPU pipeline time split 70% MLP and 30%
attention. Separately, the source-line view weights every line by the cost of
the instructions it produced. That weight is a model of instruction cost, not
a count of clock cycles. Both are useful, and neither is a share of the full
request.

**The expensive line.** Inside the MLP kernel, 82% of the weighted cost sat
on one line: the call that multiplies a tile of activations by a tile of
dequantized weights on the SIMD matrix units. The weight load and dequantize
helper next to it was 12%. In the attention kernel, two lines dominated
almost equally: the `Q·Kᵀ` score multiply (38%) and the `P·V` output
multiply (38%).

**Don't add inline rows.** The MLP kernel's wrapper, the helper it calls and
the hardware matrix intrinsic inside that helper are all reported against
the same instructions, because the compiler inlined them. Each of those rows
shows between about 81% and 100% of the kernel. Summing them would count the
same work several times. Read the innermost line that still belongs to code
you can change.

**Counters say what limits it.** Read counters only over a stretch where
the kernel you care about was the only one running; the table above uses
such windows. There the floating-point limiter was about 96% for the MLP
kernel and 91% for attention, while L1 and last-level cache limiters were all
below about 21%. The kernels are busy doing arithmetic, not waiting on
memory. So "raise occupancy" is not the fix
here; the next experiment has to reduce or restructure the multiply itself.

**A measured kernel gap is not a request gain.** The team also timed the
paged attention kernel against MLX's gather-then-attention path at the 13
attention shapes the mixed queue uses. A few of them, in milliseconds per
call:

| New rows × context | Calls in the queue | Paged kernel | MLX gather + attention |
|---|---|---|---|
| 256 × 256 | 288 | 0.97 | 0.77 |
| 256 × 1,024 | 216 | 0.98 | 1.00 |
| 20 × 1,300 | 36 | 0.46 | 0.37 |
| 60 × 1,084 | 36 | 0.40 | 0.43 |
| All 13 shapes, weighted by calls | 1,368 | 1,254.6 ms | 1,164.5 ms |

The paged kernel loses at some shapes and wins at others. Weighted by how
often each shape occurs, it was about 90 ms slower in total, roughly 0.75% of
the 12 s queue. That
is an estimate built from synchronized operator timings. It tells you where
to look, not what the request would gain.

**You do not need this view.** The reader behind it is the experimental,
version-pinned one described in Day 1, and everything you need for Week 2
works without it. The request clock and the replay are enough for every
checkpoint; the per-kernel capture, where you have it, narrows the choice to
one kernel. The source-line view only tells you which line inside that
kernel.

That completes Week 2: a tested model that serves one request at a time.
[Week 3](./week3-overview.md) starts from it and adds what serving many
requests needs: batching, chunked prefill, and paged K/V storage.

{{#include copyright.md}}
