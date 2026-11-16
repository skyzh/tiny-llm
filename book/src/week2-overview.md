<!--
  tiny-llm-book © 2022-2026 by Alex Chi Z is licensed under CC BY-NC-SA 4.0
-->

# 🚧 Week 2: A Faster Single Request

At the end of Week 1 you have a Qwen3 model that works, written in plain
array operations you can read. It is also slow, for two reasons you can see
in the code:

- every new token reruns the model over the whole prompt and everything
  generated so far; and
- every token reads every weight, expanded from the checkpoint's 4-bit codes
  to 16-bit floats, so most of each step is spent waiting on memory.

Week 2 fixes those problems one at a time, for a single request on an Apple
GPU, and lets measurements choose the order of attack. Each day starts by
profiling the model you finished the day before, changes the one part that
profile points at, and ends with a named **checkpoint** you can run, test,
time against the checkpoint before it, and profile again. By Day 6 the expensive parts run on Metal
kernels you wrote, inside the same Python model structure you built in
Week 1.

<a id="day-1-route"></a>
<a id="day-2-route"></a>
<a id="day-3-route"></a>
<a id="day-4-route"></a>
<a id="day-5-route"></a>
<a id="day-6-route"></a>

## The Route

| Day | Starting hypothesis | You build | Checkpoint | Compare with |
|---|---|---|---|---|
| [1: KV Cache](./week2-01-kv-cache.md) | Decode reruns the whole prefix | A K/V cache that grows in chunks, so decode sends only the new token. You also take your first benchmark and profile | `kv-cache` | Week 1 |
| [2: Packed W4 Weights](./week2-02-quantize-model.md) | Decode projections are limited by reading 16-bit weights | A quantized-matmul operator with a simple Metal kernel, and a model that keeps its weights in 4 bits | `packed-weights` | `kv-cache` |
| [3: Decode Matvec](./week2-03-decode-matvec.md) | One thread per output leaves the GPU idle on a single decode row | A SIMD-group matvec kernel for inputs with at most eight rows | `quantized-matvec` | `packed-weights` |
| [4: SIMD Prefill](./week2-04-simd-matrix-prefill.md) | Prefill projections still run on the simple kernel | A cooperative W4 matrix kernel for prompts and other inputs with more than eight rows | `simd-matmul` | `quantized-matvec` |
| [5: Fused Kernels](./week2-05-fused-model-kernels.md) | The small operators around the projections each make their own passes | Native RMSNorm, RoPE, and SwiGLU kernels, added one checkpoint at a time | `rmsnorm`, `rope`, `swiglu` | `simd-matmul` |
| [6: Tiled Attention](./week2-06-tiled-prefill-attention.md) | Prefill attention builds a full score matrix | Tiled attention for prompts of nine or more tokens, keeping the readable path for decode | `tiled-prefill`, then `selected` | `swiglu` |

Each starting hypothesis is something to check in your own profile before
you change anything, not a result the course promises. The days stay in this
order because each one builds on the code of the one before. If your profile
ranks things differently, write that down, do the day anyway, and compare
afterwards.

The checkpoints are cumulative. Each one keeps everything the previous one
turned on and adds one thing, so the difference between two neighbors is the
effect of that one change. `selected` is the name of the finished Week 2
model. In your solution it turns on the same features as `tiled-prefill` and
adds no new kernel of its own. (The supplied reference solution adds some
extra scheduling to `selected`; Day 6 describes it. None of it is your work.)

Pass a checkpoint to the model with `--week2-checkpoint`:

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b --max-tokens 16
```

Keep the earlier checkpoints working as you go. Each day's measurements
compare your new checkpoint with the previous one, so the older paths are
your controls.

## How Each Day Works

Every day from Day 2 on follows the same loop:

1. **Save a control.** Run the previous checkpoint on a fixed request and
   keep the result.
2. **Profile it.** Replay the same checkpoint and phase, and check whether
   the day's starting hypothesis holds on your machine.
3. **Check the operator alone.** Write the new piece, then test it against a
   readable version (a Python `mlx.core` equation or a simple kernel) on small
   inputs, using the focused test command in the chapter.
4. **Put it in the model.** Wire it into the cached model, run the day's full
   test gate, and generate text with the new checkpoint.
5. **Measure and profile again.** Time the new checkpoint against the control
   with the same model, prompt length, output length, and logit mode, and
   keep the result even if it is worse. Then replay the new checkpoint: its
   largest group is the next thing to look at.

A change can pass every test and still make the request slower, or move the
cost somewhere you did not expect. Keep that result next to the control and
write one sentence about what happened. A regression you understand is
useful; one you deleted is not.

Before you start each day, write three sentences next to your saved results:

1. what dominates, for this checkpoint and this request shape;
2. the one change you will make, and the same-workload result that would
   support it; and
3. the result that would make you reject or revert it.

For example, after `kv-cache`: "At `kv-cache:decode:128`, the
projections group dominates the replay and decode dominates the request. I
will keep projection weights packed as W4 and rerun the same request. I will
reject the change if decode time does not fall or the model's output
changes." The replay tool can record this for you with `--decision-output`
and the `--baseline-checkpoint`, `--candidate-checkpoint`,
`--dominant-category`, `--hypothesis`, `--observed-effect`, `--decision` and
`--next-experiment` fields.

[Day 1](./week2-01-kv-cache.md) shows how to run and read the benchmark and
the replay. It ends with an optional capture of GPU work, for when you want
to see inside one step; no day requires it.

The profiler and the benchmark answer different questions. The benchmark
times whole requests through your model. The profiler replays one prefill or
decode step's work through the course's public operators and your cache, one
group at a time (projections, attention, normalization and activation, and
the cache update), and reports each group's share of that replay. It builds
the work from the model weights and your public functions, so it needs no
additional model layout, TODO, or counter interface. It does use what Day 1
already asks of your cache: its offset, capacity, update behavior, and four
byte counters. It also cannot tell you whether your model actually calls
those operators; reading your wiring does that, while your tests check
behavior and the benchmark times whole requests. [Day 1](./week2-01-kv-cache.md)
explains the replay's settings.

The chapters use Qwen3-0.6B for every required comparison, so the work fits
on any Mac in the [model table](./preface.md#choose-a-model-for-your-mac).
Larger models are optional; if you try one, rerun every row you compare with
the same model.

Three kinds of results come up, and they answer different questions:

- **Your solution** (`--solution tiny_llm`) is the code you wrote.
- **The reference solution** (`--solution tiny_llm_ref`) is the course's
  completed implementation. Use it to see what a finished checkpoint should
  do. It does not fill in your unfinished functions.
- **MLX** (`--solution mlx`) runs a separate, complete MLX model. It is a
  yardstick for speed, not a part of your model.

## What the Starter Gives You

The Day 1 starter supplies model loading, the test entry points, and the
benchmark and profiling tools; you write the cache and the serving loop.
Day 2 adds the packed-weight container and the native operator declarations,
and you write the embedding, the simple kernel, and the model wiring. Days 3
to 6 each add to the kernels and the model wiring in the same way. Each
unfinished C++ entry point raises an error that names the function and the
day it belongs to, so a missing piece fails loudly instead of returning
wrong numbers.

Build both native extensions before the first Week 2 test, and rebuild after
each native change:

```bash
pdm run build-ext
pdm run build-ext-ref
```

## Where Week 2 Ends Up

How close does the finished Week 2 model get to MLX? The supplied reference
solution, with all of Week 2's kernels in place plus some extra scheduling, was timed
against stock MLX on Qwen3-4B for eight prompt and output lengths, from
128 / 128 to 32,640 / 128 tokens. Measured as total throughput for a whole
request, it reached between 95.1% and 101.3% of MLX's speed. It was faster
than MLX only in the longest-output case (8,192 / 8,192).

That summary hides real costs:

- it took longer than MLX to produce the first token in every case;
- decode was faster in two cases and slower in the rest;
- its peak allocated memory was higher in seven of the eight cases, by up to
  about 0.9 GB; and
- at 128 / 128 and 8,192 / 128, several individual paired runs fell below 95%
  even though the medians did not.

These numbers come from separate measurement runs on one Apple-silicon Mac,
made on a development revision before the reference was merged, not from a
fresh run of this checkout. The full table, the method, and the remaining
limits, including numerical differences from MLX and answer-quality checks,
are in [PR #328](https://github.com/skyzh/tiny-llm/pull/328). Your own runs
with 0.6B will give different numbers. The value of the course model is that
you can see and change every part of it, not that it beats MLX.

<a id="historical-lessons-and-week-3"></a>

## Older Week 2 Material

An earlier version of Week 2 had seven days in a different order. Those
lessons still open at their old addresses, but they are no longer listed in
the sidebar: some of their commands no longer exist, and their code differs
from Days 1 to 6. The
[performance appendix](./appendix-performance.md) still records results from
that older code, so treat it as history rather than as numbers for this
checkout.

Week 2 also had five days until recently, with the decode matvec inside Day 2
and a second, preallocated cache checkpoint called `capacity-cache` on Day 1.
The old addresses of the five-day Days 3 to 5 now open the matching chapter.
`capacity-cache` still works as another name for `kv-cache`, and the old test
numbering still runs with
`pdm run test --week 2 --day N --legacy-week2-days`. If you copied a day's
tests under the old numbering, refresh them with
`pdm run copy-test --week 2 --day N --force` before running the new day.

## After Week 2

[Week 3](./week3-overview.md) starts from the dense cached model you finish
here and turns it into a server: continuous batching, chunked prefill, paged
K/V storage, and attention that reads pages directly.

{{#include copyright.md}}
