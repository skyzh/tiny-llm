# 🚧 Week 2: Find the Next Bottleneck

After [Day 1](./week2-01-kv-cache.md) you have a model that reuses its prefix
and keeps its cache inside a fixed bound. It works, and it is faster than
Week 1. The rest of Week 2 makes it faster again, but not by working through
a list of kernels. Each day starts from a profile of the model you have right
now, picks the one thing that profile says is expensive, changes only that,
checks the answer is still right, and measures again.

This chapter teaches that loop and the tools behind it. Read it once after
Day 1, then come back to it at the start of every later day.

## The Loop

Every day from Day 2 on follows the same six steps:

1. **Run the baseline.** Save a benchmark of the current checkpoint before you
   change anything. This is the number you will compare against.
2. **Measure where the time goes.** Profile the same checkpoint on the same
   request shape.
3. **Write down a mechanism.** Not "attention is slow", but "decode reads
   every projection weight once per token, so its time should scale with
   weight bytes".
4. **Make one learner-owned change.** Only the change the day asks for.
5. **Check it.** Run the day's tests first. Then rerun the benchmark on the
   same request and compare latency and memory with the saved baseline.
6. **Keep the result, even if it is worse.** Then profile the changed model.
   That profile, not the course outline, tells you what to look at next.

Step 6 matters more than it looks. A change can pass every test and still make
the request slower, or move cost somewhere you did not expect. Save that
result next to the baseline and write one sentence about what happened. A
regression you understand is useful; one you deleted is not.

## Four Ways to Look at One Request

A "profile" can mean four quite different measurements. They answer different
questions, and the most common mistake is to read one as if it were another.

| View | What it measures | Answers | Cannot answer |
|---|---|---|---|
| Request clock | Wall time for a whole request: time to first token, decode speed, peak memory | Did the request get faster or use less memory? | Which operator caused it |
| System timeline | CPU and GPU activity on one shared clock | Is the GPU idle while Python prepares work? Do CPU and GPU overlap? | How expensive one kernel is inside |
| Operator replay | Synchronized, repeated timing of groups of operators at one fixed shape | Which group of operators costs the most at this shape? | Whether your model actually calls them, or what a full request costs |
| GPU capture, counters and source lines | One captured stretch of GPU work, with per-kernel cost, hardware limiters and weighted cost per source line | Which kernel dominates, what limits it, which line inside it is expensive | Anything about the CPU, the full request, or other shapes |

Read the table from top to bottom when you start, and from bottom to top when
you finish:

- **Start at the top.** The request clock tells you whether there is a problem
  at all and which phase it lives in. If decode is already fast and prefill is
  slow, a decode kernel cannot be your next change.
- **Narrow down.** The timeline tells you whether the time is GPU work at all.
  Replay and capture then tell you which operator and which kernel.
- **Finish at the top again.** After a change, the only number that tells you
  the request improved is the request clock. A faster kernel inside a request
  that waits on the CPU may not move it at all.

The rest of this chapter goes through each view on your `capacity-cache`
checkpoint.

## View 1: The Request Clock

Start with the benchmark you already ran at the end of Day 1. It runs each
variant in fresh processes, alternates their order, and reports prefill and
decode separately. `--offline` needs the model files cached already:

```bash
pdm run build-ext
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 \
  --repeats 2 --variant week2-capacity-cache --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-baseline-capacity-cache.json
```

Keep this file. Every later day compares against a saved result like it, on
the same model, prompt length, output length and logit setting. The `mlx` row
is a separate full model, there to show scale; it is not your control.

Look at three things:

- **Which phase is larger.** With 128 prompt tokens and 129 output tokens,
  one prefill pass produces the first token and 128 one-token decode steps
  produce the rest. If decode dominates total time, the next change should
  make each decode step cheaper.
- **Decode tokens per second.** One decode step sends one token through every
  layer. For a 0.6B model that is little arithmetic, but every projection
  weight still has to be read from memory once per step. If the step time is
  close to "weight bytes divided by memory bandwidth", the step is limited by
  reading weights, not by arithmetic.
- **Peak memory.** Note it now. Day 2 changes how weights are stored, so
  check it again there. If a later change raises it, that is a cost you must
  report.

Two short runs on a busy laptop can differ by several percent. If two
variants are within that noise, rerun them before you draw a conclusion, and
close other GPU-heavy programs first.

## View 2: Operator Replay

The request clock says decode is expensive. Operator replay says which part of
a decode step is expensive. It seeds a cache, then times groups of operators
(projections, attention, normalization and position, the cache update)
separately at one fixed shape, synchronizing the GPU around each group:

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case capacity-cache:decode:128 --warmup 4 --iterations 12 \
  --json-output week2-capacity-cache-decode-128.replay.json
```

[Day 1's profiler section](./week2-01-kv-cache.md#attribute-the-cached-model)
explains the case name and every output field. The part that matters here is
how to read it:

- **Read the largest group, not the total.** The replay total is the sum of
  each group's median. It is a useful denominator for shares, but it is not
  the time of a real decode step: a real step does not synchronize between
  groups, so the GPU can overlap work the replay keeps apart.
- **A share is not a prediction.** If projections are 70% of the replay, a
  projection twice as fast does not make the request 35% faster. Only the
  request clock can tell you that.
- **The replay does not prove wiring.** It calls the operators directly. It
  cannot see whether your model's forward pass uses them. Day 2's wiring check
  and the model tests answer that.

Open the JSON file. Each complete row names its `checkpoint` and has a
`workload` block describing what was replayed: the phase, context length
`S`, cache capacity `C` and logit setting. Then come the per-group medians
and shares, the `replay_total_us` they are shares of, and a `cache` block
with offsets, capacity and Day 1's byte counters. Two replays can only be
compared when their `workload` blocks match. Change the checkpoint and
nothing else.

Some rows are not complete. A case the tool cannot replay reports
`coverage.complete` as false, with a reason and no total. Treat that as a
missing measurement, never as zero.

## View 3: The System Timeline

A replay hides one important question: is the GPU even busy? During decode,
Python prepares each small step, MLX records it, and the GPU runs it. If the
GPU finishes a step before the CPU has the next one ready, it sits idle, and
a faster kernel does nothing about that idle time.

A system timeline puts CPU samples and GPU work on one clock, so you can see
the gaps. The course does not supply a timeline command yet; the case study
below shows what one looks like. You do not need it for any Week 2
checkpoint. When you cannot record one, keep two rules in mind: a request
whose decode steps are very short is the one most likely to be waiting on the
CPU, and a kernel improvement that the replay sees but the request clock does
not is a hint that the GPU was not the limit.

## View 4: GPU Capture, Counters and Source Lines

The deepest view captures one synchronized stretch of GPU work and replays it
under Apple's GPU profiler. It is optional, needs macOS 27 with
`/usr/bin/gpudebug`, and is never a gate for a checkpoint. Run the focused
tests first: there is no point profiling a kernel that gives wrong answers.

Create `out/` first and choose file names that do not exist yet. The helper
refuses to overwrite a trace, metadata file or manifest:

```bash
mkdir -p out

MTL_CAPTURE_ENABLED=1 pdm run capture-week2 \
  --solution tiny_llm --model qwen3-0.6b \
  --checkpoint capacity-cache --phase decode --tokens 128 \
  --prefill-logits last \
  --trace out/capacity-cache-decode-128.gputrace \
  --metadata out/capacity-cache-decode-128.capture.json \
  --manifest out/capacity-cache-decode-128.trace-manifest.sha256

gpudebug --json -t out/capacity-cache-decode-128.gputrace --timeout 1800 \
  -c 'profile run --gpu-state default --exec serial' \
  > out/capacity-cache-decode-128.profile.jsonl

pdm run reduce-week2-gpudebug \
  --capture-metadata out/capacity-cache-decode-128.capture.json \
  --manifest out/capacity-cache-decode-128.trace-manifest.sha256 \
  --profile-jsonl out/capacity-cache-decode-128.profile.jsonl \
  --output out/capacity-cache-decode-128.gpudebug.json
```

The capture helper compiles and warms the shape first, so the trace holds one
steady-state region rather than start-up work. Its logit setting defaults to
`all` for older captures, so pass `--prefill-logits last` to match the
benchmark and replay above. The cache bound defaults to the context length,
as in the replay. Trace packages can be large:
check free disk space before you start. Once you have kept the reduced JSON,
you can delete the `.gputrace` package and raw `.jsonl` streams, unless you
also want to try the optional source-line reader below: it reads the raw
trace and its companion files, and the reduced JSON cannot replace them, so
keep them until that read is finished.

The reduced file can contain three kinds of information. Any of them may be
missing on your machine; a missing tree stays "unavailable", never zero.

- **Per-kernel cost.** Each kernel's share of GPU shader time in the captured
  region. This is the first thing to read: one kernel name usually dominates.
- **Hardware counters.** Limiters and utilization for units such as the
  floating-point ALUs, L1 cache and last-level cache. A limiter near 100%
  says that unit is the bottleneck. Low occupancy alone is not a diagnosis;
  check the limiters before you blame it.
- **Source-line cost.** Apple's shader cost view can weight each line of
  Metal source by the instructions it produced. The reducer above does not
  produce it. See the end of this section for an optional reader and its
  limits.

### Optional: Read Source-Line Costs

The course supplies an advanced reader for this view:

```bash
pdm run source-costs-week2 CAPTURE.gputrace --out NEW_DIRECTORY
```

It writes CSV and JSON files that list source locations and their weighted
costs, without copying any source text. Its input is narrower than the
capture above, and that is the important part:

- it needs an existing `.gputrace` that already holds sampled shader-profile
  data, with exactly one profiler session inside it and its companion
  library and symbol files;
- keep `sources-manifest.json` and a `sources/` directory beside the trace if
  you want the source locations marked as verified; without them they are
  reported as unverified;
- it reads Apple's profiler data through a private framework. The decoder
  behind it was exercised only on one machine with macOS 27.0.1 on Apple
  silicon, Python 3.12.13 and Xcode 27.2. This course command wraps that
  decoder and has not yet been run against a real capture. A different Xcode
  can change or break it.

It is not established that a trace made with `capture-week2` and
`gpudebug … profile run` contains the samples this reader needs. Treat it as
an experiment, not as part of the Week 2 route. The case study below shows
what the view adds.

## If You Cannot Capture

Many setups will not have `gpudebug`, and not every capture exposes every
tree. That does not block anything in Week 2. The request clock and the
operator replay only need the environment from
[Setting Up the Environment](./setup.md): an Apple silicon Mac with the
course's PDM setup and the cached model. Together they answer the two
questions every day needs: did the request get faster, and which group of
operators changed?

To practise reading a capture without making one, open the compact example
the repository ships at
`benchmark_results/m4-pro-qwen3-4b-week2-gpudebug-macos27-mlx-0.32.0.json`.
It comes from an earlier version of this course, on an M4 Pro with
Qwen3-4B, so its checkpoint names and kernels will not match yours. Look at
how it is organized instead. It records which source, host, software and
fixed workload produced it. Each capture lists what was available and what
was not. Each `decisions` entry names the change, what dominated before, the
evidence and whether the change was kept. That last part is the shape your
own notes should have.

## Profile Again After the Change

Step 6 of the loop is the one people skip. After Day 2, rerun the benchmark
and the replay on the new checkpoint, changing only the checkpoint name (and
the capture too, if you made one):

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 \
  --repeats 2 --variant week2-capacity-cache --variant week2-quantized-matvec \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-after-quantized-matvec.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case quantized-matvec:decode:128 --warmup 4 --iterations 12 \
  --json-output week2-quantized-matvec-decode-128.replay.json
```

Putting both checkpoints in one benchmark run lets the runner alternate them,
so a slow minute on your machine does not land on only one side. Then compare
the two replays group by group. Three outcomes are common, and each one
tells you something:

- **The target group fell and the request got faster.** The mechanism you
  wrote down survived a test that could have disproved it. Note how much of
  the replay change showed up in the request; it is rarely all of it.
- **The target group fell but the request did not move.** Something else was
  the limit: another group, the CPU, or prefill rather than decode. Check the
  phase split in the request clock before you go further.
- **The request got slower.** Keep the result. Check that the model still
  calls the new path, then look for the group that grew. A new kernel can be
  faster at the replay shape and slower at the shapes the real request uses.

Whichever happened, the new replay's largest group is your next candidate.

## Case Study: Reading a Whole Request

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

## Case Study: Inside One Kernel

The second case study goes one level deeper, into the prefill MLP kernel
and a paged-attention kernel, captured once each at one representative shape
(a 256-row prefill chunk). It shows what the source-line view adds, and also how
easily it is misread.

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

**You do not need this view.** The reader behind it is the optional,
version-pinned one described earlier, and everything you need for Week 2
works without it. The request clock and the replay are enough for every
checkpoint; the per-kernel capture, where you have it, narrows the choice to
one kernel. The source-line view only tells you which line inside that
kernel.

## Write the Decision Down

Before you start the next day, write three sentences next to your saved
results:

1. what dominates, for this checkpoint and this request shape;
2. the one change you will make, and the same-workload result that would
   support it;
3. the result that would make you reject or revert it.

For example, after `capacity-cache`: "At `capacity-cache:decode:128`, the
projections group dominates the replay and decode dominates the request. I
will keep projection weights packed as W4 and rerun the same request. I will
reject the change if decode time does not fall or the model's output
changes." The replay tool can record this for you with `--decision-output`
and the `--baseline-checkpoint`, `--candidate-checkpoint`,
`--dominant-category`, `--hypothesis`, `--observed-effect`, `--decision` and
`--next-experiment` fields.

## Route Through Week 2

The course gives the days in a fixed order, because each change needs the
ones before it. The W4 matrix kernel in Day 3 reuses the packed weights from
Day 2, and Day 5's attention runs inside the fused model from Day 4. That
order is a reasonable route, not a prediction of what your profile will show.
At the start of each day, profile first and check whether the day's target
really is the expensive part on your machine. If it is not, still do the day
(the later days build on it), but write down what you saw instead.

| Start from | Hypothesis to check in your profile | Change | Checkpoint |
|---|---|---|---|
| Week 1 model | Decode time grows with the prefix, because every step reruns all earlier tokens | Reuse earlier keys and values | `kv-cache` |
| `kv-cache` | Cache storage grows by concatenation on every step | Allocate once from the request bound | `capacity-cache` |
| `capacity-cache` | Decode is dominated by projections, which read every weight once per token | Keep weights packed as W4 and decode with a matvec | `quantized-matvec` (Day 2) |
| `quantized-matvec` | Prefill projections use the readable fallback for prompts longer than eight rows | Cooperative SIMD matrix kernel | `simd-matmul` (Day 3) |
| `simd-matmul` | Small operators around the projections each make their own launch and memory pass | Fused RMSNorm, then RoPE, then SwiGLU | `rmsnorm`, `rope`, `swiglu` (Day 4) |
| `swiglu` | Prefill attention builds an `L × S` score matrix per head | Tiled attention with a running softmax | `tiled-prefill`, `selected` (Day 5) |
| `selected` | Whatever is left; in the case studies below, the MLP matrix multiply | Carry the question into Week 3 | — |

Each entry in the middle column is a hypothesis for you to test, not a
result the course promises. On a different chip, a different model size or a
different prompt length, your profile can rank things differently. When it
does, the profile wins: write down what you measured, do the day anyway
because later days build on it, and compare again afterwards.

{{#include copyright.md}}
