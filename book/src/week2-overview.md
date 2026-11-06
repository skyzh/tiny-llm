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
GPU. Each day starts from the model you finished the day before, changes one
part of it, and ends with a named **checkpoint** you can run, test, and time
against the checkpoint before it. By Day 5 the expensive parts run on Metal
kernels you wrote, inside the same Python model structure you built in
Week 1.

<a id="day-1-route"></a>
<a id="day-2-route"></a>
<a id="day-3-route"></a>
<a id="day-4-route"></a>
<a id="day-5-route"></a>

## The Route

| Day | You build | Checkpoint | Compare with |
|---|---|---|---|
| [1: Cache and Measure](./week2-01-kv-cache.md) | A K/V cache, so decode sends only the new token; then a cache sized from the request's limit | `kv-cache`, then `capacity-cache` | Week 1, then `kv-cache` |
| [2: Keep W4 Packed](./week2-02-quantize-model.md) | A quantized-matmul operator with two Metal kernels, and a model that keeps its weights in 4 bits | `quantized-matvec` | `capacity-cache` |
| [3: SIMD Matrix Prefill](./week2-03-simd-matrix-prefill.md) | A cooperative W4 matrix kernel for prompts and other inputs with more than eight rows | `simd-matmul` | `quantized-matvec` |
| [4: Fused Model Primitives](./week2-04-fused-model-kernels.md) | Native RMSNorm, RoPE, and SwiGLU kernels, added one checkpoint at a time | `rmsnorm`, `rope`, `swiglu` | `simd-matmul` |
| [5: Tiled Dense Prefill Attention](./week2-05-tiled-prefill-attention.md) | Tiled attention for prompts of nine or more tokens, keeping the readable path for decode | `tiled-prefill`, then `selected` | `swiglu` |

The checkpoints are cumulative. Each one keeps everything the previous one
turned on and adds one thing, so the difference between two neighbors is the
effect of that one change. `selected` is the name of the finished Week 2
model. In your solution it turns on the same features as `tiled-prefill` and
adds no new kernel of its own. (The supplied reference solution adds some
extra scheduling to `selected`; Day 5 describes it. None of it is your work.)

Pass a checkpoint to the model with `--week2-checkpoint`:

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint capacity-cache --model qwen3-0.6b --max-tokens 16
```

Keep the earlier checkpoints working as you go. Each day's measurements
compare your new checkpoint with the previous one, so the older paths are
your controls.

## How Each Day Works

Every day follows the same loop:

1. **Save a control.** Run the previous checkpoint on a fixed request and
   keep the result.
2. **Check the operator alone.** Write the new piece, then test it against a
   readable version (a Python `mlx.core` equation or a simple kernel) on small
   inputs, using the focused test command in the chapter.
3. **Put it in the model.** Wire it into the cached model, run the day's full
   test gate, and generate text with the new checkpoint.
4. **Measure.** Time the new checkpoint against the control with the same
   model, prompt length, output length, and logit mode.

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
benchmark and attribution tools; you write the cache and the serving loop.
Day 2 adds the packed-weight container and the native operator declarations,
and you write the embedding, the kernels, and the model wiring. Days 3 to 5
each add the declarations for that day's kernels in the same way. Each
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
solution, with all five days in place plus some extra scheduling, was timed
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
lessons stay at their old addresses as
[historical Week 2 material](./week2-02-benchmark-profile.md). They cover
benchmark method, the W4 derivation, bandwidth and roofline calculations for
Apple M1 to M4 chips, fusion and SIMD mechanisms, optional capture with
macOS GPU tools, and the old bounded-decode and Split-K experiments. Their
[operator-attribution diagram](./week2-kernel-profile.svg) and
[decision diagram](./week2-performance-summary.svg) describe that older code.
Some of their commands no longer exist. Use Days 1 to 5 above for the
current course.

The [performance appendix](./appendix-performance.md) also records results
from that older code, so treat it as history rather than as numbers for this
checkout.

## After Week 2

[Week 3](./week3-overview.md) starts from the dense cached model you finish
here and turns it into a server: continuous batching, chunked prefill, paged
K/V storage, and attention that reads pages directly.

{{#include copyright.md}}
