# 🚧 Week 2 Day 3: Decode Matvec

[Day 2](./week2-02-quantize-model.md) kept the weights packed, but every
projection still runs on the simple kernel, with one thread per output. On a
decode step, with a single activation row, each of those threads works
through one long dot product on its own while most of the GPU waits.

Start from your `packed-weights` replay. If the projections group still leads
the decode step, today's hypothesis fits your machine: the packed bytes are
being read by too few threads, and spreading each dot product across a SIMD
group should shrink that group. If another group leads, write down which one,
do the day anyway, and compare afterwards.

Today you write a second kernel for small row counts and route decode to it.
Prompts with more than eight rows stay on the simple kernel until
[Day 4](./week2-04-simd-matrix-prefill.md). The new checkpoint is
`quantized-matvec`.

```bash
pdm run build-ext
pdm run build-ext-ref
pdm run test --week 2 --day 3
```

The day's tests fail until you finish the tasks below.

## Task 1: Write the Decode Matvec

```
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

Implement `quantized_matvec_x4_fast_w4a16_g128` in `quantized_matmul.metal`,
instantiated for `bfloat16_t`. It computes the same math as Day 2's simple
kernel, shaped for `M <= 8` rows. Then launch it from
`QuantizedMatmul::eval_gpu` in `quantized_matmul.cpp`, as described at the
end of this task.

With `M = 1`, there is only one activation row, and each output is a dot
product of length `N`, which is 2,560 for Qwen3-4B's hidden size. Giving that
whole product to one thread leaves most of the GPU idle. Instead, give one
output tile to a SIMD group:

- each lane takes a different slice of the `N` inputs and keeps a partial
  sum;
- at the end, `simd_sum` adds the 32 partial sums together;
- lane 0 writes the result.

That works for one output, but every lane still loads activations that the
neighboring output will load again. So let each SIMD group compute **four**
outputs at once. In the reference schedule, each lane loads two adjacent
packed words of a row, which is 16 activations, and reuses those 16 activations
for all four output rows. Two SIMD groups per threadgroup gives eight outputs
per threadgroup. For `M` rows and `K` outputs, the grid has
`M × ceil(K / 8)` threadgroups of 64 threads each.

Two more tricks make the inner loop cheaper.

**Apply the bias once per group.** Use the identity from
[Day 2's math section](./week2-02-quantize-model.md#the-math-you-will-implement).
Each lane keeps `Σ a·q` and `Σ a` separately, then adds
`scale * Σ a·q + bias * Σ a`. The bias is applied once per group instead of
once per weight.

**Skip the shifts.** Four codes fit in one 16-bit half-word. Masking that
half-word with `0x00F0` gives `16 * q` instead of `q`. If you divide the
matching activation by 16 once, before the loop, the factor cancels. Call the
pre-divided activations `a0'` to `a3'`:

```plain
hw  = 0x7620                      # codes 0, 2, 6, 7
a0' = a0,  a1' = a1/16,  a2' = a2/256,  a3' = a3/4096

a0' * (hw & 0x000F) = a0        * 0     = a0 * 0
a1' * (hw & 0x00F0) = (a1/16)   * 32    = a1 * 2
a2' * (hw & 0x0F00) = (a2/256)  * 1536  = a2 * 6
a3' * (hw & 0xF000) = (a3/4096) * 28672 = a3 * 7
```

The kernel scales each lane's 16 activations once, then uses only masks and
multiply-adds for all four outputs. Fewer integer instructions do not
guarantee a faster kernel, though. The extra live values compete for
registers. Judge the trick by measuring the whole model, as described below.

A lane's 16 values come from two adjacent words, and a group of 128 spans 16
words, so a lane's slice never straddles two groups. That is why one scale and
one bias per output per loop iteration are enough.

Read activations straight from device memory. One activation row is small and
stays in cache. Copying it into threadgroup memory first would add a barrier
to every projection. If you try that as an experiment, keep it only if the
whole-model measurement improves.

### Launch It from `eval_gpu`

Extend `QuantizedMatmul::eval_gpu` in `quantized_matmul.cpp`: when the
binding's `use_simdgroup` flag is true and `M <= 8`, launch the matvec kernel
with a grid of `M × ceil(K / 8)` threadgroups of 64 threads; otherwise keep
Day 2's simple kernel.

Day 2's `quantized_matmul(..., use_simdgroup=True)` in `quantize.py` already
passes that flag through, so you can check the kernel before writing any new
Python:

```bash
pdm run build-ext
pdm run test --week 2 --day 3 -- -k task_1
```

The `task_1` test calls `quantized_matmul` with `use_simdgroup=True` on a
decode shape and compares the result with your simple kernel.

## Task 2: Route Small Row Counts to It

```
src/tiny_llm/quantize.py
```

Implement `quantized_matvec_custom` in `quantize.py`. It calls the extension with the default `use_simdgroup` and rejects inputs with
more than 8 rows. Then update `quantized_linear`: count the rows, and call
`quantized_matvec_custom` when there are at most 8 and `w.use_simdgroup_matvec`
is set; otherwise call `quantized_matmul` as on Day 2.

```bash
pdm run test --week 2 --day 3 -- -k task_2
```

These tests compare the matvec path with `mx.quantized_matmul` at `M = 1`,
at `M = 8`, and at a real Qwen3-4B projection shape (one row of 2,560
inputs, 1,024 outputs). The full Day 3 gate also runs output counts that do not fill
the last tile (5 and 65 outputs).

### Optional: Tune the Schedule

The four-output, two-SIMD-group layout is a measured starting point, not a
rule. If you want to see why, change one choice at a time and measure the
whole model each time:

- **Outputs per SIMD group:** try 2, 4, and 8. More outputs reuse each
  activation load more often but keep more accumulators alive, which uses
  registers.
- **SIMD groups per threadgroup:** try 2, 4, 8, and 16. More groups cover more
  outputs per threadgroup but can lower how many threadgroups fit on a core.

Adding SIMD groups to a threadgroup and adding threadgroups to the grid are
different knobs. The first changes per-threadgroup register demand. The
second changes how the outputs are divided. Keep the experiments separate so
each number answers one question. If you add an experimental variant, keep
the selected path's kernel names unchanged.

## Task 3: Use It in the Model

```
src/tiny_llm/qwen3_week2.py
```

In `Qwen3ModelWeek2.__init__`, load the packed weights with
`use_simdgroup_matvec` taken from the checkpoint's `simdgroup_matvec` feature.
It is false for `packed-weights` and true from `quantized-matvec` onward, so
`packed-weights` stays a fair control. Nothing else in the model changes.

```bash
pdm run test --week 2 --day 3

pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-0.6b --max-tokens 16
```

## Check That the Model Really Uses Your Kernel

Make sure that:

- a decode step goes `quantized_linear` → `quantized_matvec_custom` → your
  primitive → the matvec kernel; and
- a prompt longer than eight tokens goes `quantized_linear` →
  `quantized_matmul` → your primitive → the simple kernel.

An easy way to check is to put a temporary exception in one branch of
`quantized_linear` and confirm that the `main` command above hits it.

Then measure against Day 2, with the same model, lengths and logit mode:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-packed-weights --variant week2-quantized-matvec --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day3-matvec.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case packed-weights:decode:128 --case quantized-matvec:decode:128 \
  --warmup 4 --iterations 12 \
  --json-output week2-day3-attribution.json
```

Ask the same three questions as on Day 2: did the projections group fall,
did the request get faster, and does the model really call the new path?
Keep the result whatever it is.

Then look for the next bottleneck. Decode now runs your matvec, but a prompt
with more than eight rows still goes to the simple kernel. Replay one prefill
as well:

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case quantized-matvec:prefill:128 --warmup 4 --iterations 12 \
  --json-output week2-day3-prefill.json
```

If the projections group leads that prefill replay, you have Day 4's starting
point. If something else does, note it before you go on.

For a sense of scale, here is one older measurement of packing the weights and
decoding with a matvec together. It used Qwen3-4B on an M4 Pro, with an
earlier version of the course code and an earlier version of the profiler
whose numbers are not comparable with today's replay. The time spent in
projections fell by 69.0%, and decode on a fixed workload went from 24.38 to
58.90 tokens per second. That older profile then showed normalization,
positional encoding, and activation taking 33.5% of the time, which is what
Day 5 goes after. Your 0.6B numbers will differ, and these are not targets to
hit. The [performance appendix](./appendix-performance.md) has the full
history.

To compare at 4B yourself, download the model with
`hf download Qwen/Qwen3-4B-MLX-4bit` and rerun **every** row with
`--model qwen3-4b`.

{{#include copyright.md}}
