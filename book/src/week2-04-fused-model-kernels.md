# 🚧 Week 2 Day 4: Fused Model Primitives

After [Day 3](./week2-03-simd-matrix-prefill.md), the projections run on your
own packed-weight kernels for both decode and prefill. Profile a decode step
now and a different set of operators shows up. Every layer still runs RMSNorm
(four times: before attention, on the queries, on the keys, and before the
MLP), RoPE (twice, on the queries and keys), and SwiGLU (once, in the MLP),
all written as the readable Week 1 `mlx.core` equations. On one older M4 Pro
run with Qwen3-4B, once the projections were packed, normalization,
positional encoding, and activation together took about a third of the
profiled decode time.

Each of those operators does very little arithmetic. They are slow because of
how the work is described: several small array operations, each reading and
writing a full tensor, for a few multiplies per element. Today you replace each
one with a single Metal kernel, one checkpoint at a time:

| Checkpoint | Adds |
|---|---|
| `rmsnorm` | Your RMSNorm kernel in every norm of the model |
| `rope` | Your RoPE kernel, on top of `rmsnorm` |
| `swiglu` | Your SwiGLU kernel, on top of `rope` |

The model, the cache, and `quantized_linear` stay the same.

## Save a Control First

Time the cached 0.6B `simd-matmul` checkpoint before you change anything. The
`--offline` run needs the model cached already. The `mlx` row is a separate
complete model, there for scale:

```bash
pdm run build-ext
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-simd-matmul --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day4-control.json
```

Today's files already exist in the starter:

```plain
src/tiny_llm/week2_kernels.py
src/extensions/src/week2_kernels.cpp
src/extensions/src/week2_kernels.metal
```

The header declarations, Python bindings, and CMake entries are in place too,
and each unfinished C++ function raises an error naming itself. Build the
reference extension to see the finished behavior, then run the first focused
test, which should stop at the `FastRMSNorm` TODO:

```bash
pdm run build-ext-ref
pdm run test-refsol --week 2 --day 4
pdm run test --week 2 --day 4 -- -k task_1
```

For each operator, check its numbers against a readable version first, then
run the live model at that operator's checkpoint. The live run checks that the
whole model still works with the change. Neither check shows which kernel the
model actually called, because the readable path gives the same answers; for
that, read your wiring (Task 4). For RMSNorm you can also look at its
optional dispatch counts.

<a id="why-fusion-helps"></a>

## Why One Kernel Beats Several Operations

Take RMSNorm. As `mlx.core` operations it is roughly: cast to float, square,
take the mean of each row, add epsilon, take the reciprocal square root,
multiply, cast back, and multiply by the learned weight. MLX can fuse some
neighbouring element-wise steps, but the row reduction splits the work, so
the graph may still launch several GPU operations and write intermediate
tensors between them.

One Metal kernel can do the whole thing in a single launch:

- each input element is read once and kept in registers;
- the sum of squares is reduced across lanes without going through memory;
- accumulation happens in float where precision needs it;
- only the final output is written; and
- the launch grid is shaped for the rows the model actually has.

MLX still schedules the operation, allocates its buffers, and launches your
Metal function through the extension API. You own the arithmetic inside it.
Because MLX may already fuse part of the readable version, don't assume a
gain: measure it on the same request.

## Task 1: RMSNorm

Start with `tiny_llm_ext::rms_norm` and `Week2RMSNorm::eval_gpu` in
`src/extensions/src/week2_kernels.cpp`. Add Metal
`week2_rms_norm_register_cached` and a readable `week2_rms_norm` fallback in
`src/extensions/src/week2_kernels.metal`, then implement
`FastRMSNorm.__call__` in `src/tiny_llm/week2_kernels.py`. Keep the supplied interfaces. Make
`Week2RMSNorm::eval_cpu` in `src/extensions/src/week2_kernels.cpp` raise a
GPU-only error instead of the starter TODO.

The equation:

```plain
inverse_rms = rsqrt(mean(x[i]²) + epsilon)
output[i]   = x[i] * inverse_rms * weight[i]
```

The interesting part is the row sum. A Qwen3-4B hidden row has 2,560 values.
One 32-lane SIMD group would leave each lane about 80 values to add up one
after another. Spread the row over more threads instead.

**The fallback kernel** uses 256 threads (eight SIMD groups) per row. Each
group adds up its share with `simd_sum`, lane 0 of each group writes that
partial sum to threadgroup memory, and the first group adds the eight
partials:

```plain
group_sum   = simd_sum(each lane's local sum of squares)
row_sum     = simd_sum(the groups' partial sums)
inverse_rms = rsqrt(row_sum / hidden_size + epsilon)
```

Each thread then reads its inputs again to write the output.

**The register-cached kernel** avoids that second read. Each thread holds up
to four input values in registers through the whole reduction, then scales
and writes them directly. A 2,560-value row needs 640 threads (2,560 / 4,
rounded up to a whole SIMD group). Use this kernel for rows up to 4,096 wide
and the fallback above that. The supplied `FastRMSNorm.__init__` creates a
`dispatch_counts` dictionary with `register_cached` and
`fixed_width_fallback` entries. Incrementing the matching entry on each call is
optional, but it gives you a quick way to see which path ran. No test or tool
requires it.

In the primitive, check shape and dtype, allocate the output, bind buffers
and constants, and give each active SIMD group one float slot for its partial
sum. Instantiate the BF16 kernel. Accumulate the sum and compute the scale in
float, and cast once at the end. The readable version may round at a
different step, so compare with a tolerance, not exact equality.

Use `FastRMSNorm` for every norm in the Week 2 model: query, key, both layer
norms, and the final norm. Then check the operator and run the model before
starting RoPE:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k task_1
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint rmsnorm --model qwen3-0.6b --max-tokens 16
```

The test compares your BF16 output with `mx.fast.rms_norm` on a narrow,
16-value row, so it exercises only your register-cached path. The repository
also has tests for rows at and past the 4,096 limit
(`tests_refsol/test_rmsnorm_register_cache.py`), but they import the
reference solution, not yours, and the Day 4 test command does not run them
against your code. Check your wide-row fallback yourself, for example by
comparing a 4,097-wide row with the readable equation.

## Task 2: RoPE

Next, complete `tiny_llm_ext::rope` and `Week2RoPE::eval_gpu` in
`src/extensions/src/week2_kernels.cpp`, add the `week2_rope` function in
`src/extensions/src/week2_kernels.metal`, and finish `FastRoPE.__init__` and
`FastRoPE.__call__` in `src/tiny_llm/week2_kernels.py`. Make
`Week2RoPE::eval_cpu` in `src/extensions/src/week2_kernels.cpp` raise a
GPU-only error instead of the starter TODO.

RoPE rotates each pair of values in a head by an angle that depends on the
token's position and the pair's index:

```plain
angle = (batch_offset + token_position) * base ** (-pair / (dims / 2))
real' = real * cos(angle) - imag * sin(angle)
imag' = imag * cos(angle) + real * sin(angle)
```

The angle does not depend on the head. A simple kernel with one thread per
element computes the same angle, sine, and cosine twice for each pair, and
again for every head. Instead, give each thread one pair index and a block of
four heads: compute the angle, sine, and cosine once, then rotate that pair in
all four heads. Work directly in the model's `B, L, H, D` layout, and use
Metal's `fast::exp2`, `fast::sin`, and `fast::cos` for BF16.

Accept either one offset for the whole batch or one offset per batch row, and
turn both into an int32 array before dispatch. Per-row offsets matter in
Week 3, when requests at different positions share a batch. Build that offset
array once per model call, outside the layer loop, rather than once per layer.

Compare this with the readable version, which builds position arrays, looks
up sine and cosine tables, splits the head in two, runs several element-wise
operations, and concatenates the halves. The kernel reads each pair once and
writes each rotated value once.

Swap it into the model. The `rope` checkpoint keeps your RMSNorm. Check the
operator, then run the model before starting SwiGLU:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k task_2
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint rope --model qwen3-0.6b --max-tokens 16
```

The test checks BF16 output with per-row offsets.

## Task 3: SwiGLU

Finish with `tiny_llm_ext::swiglu` and `Week2SwiGLU::eval_gpu` in
`src/extensions/src/week2_kernels.cpp`. Add the `week2_swiglu` function in
`src/extensions/src/week2_kernels.metal`, and `swiglu` in
`src/tiny_llm/week2_kernels.py`. Make `Week2SwiGLU::eval_cpu` in
`src/extensions/src/week2_kernels.cpp` raise a GPU-only error instead of the
starter TODO.

SwiGLU combines the MLP's gate and up branches:

```plain
output = (gate / (1 + exp(-gate))) * up
```

This one is simple: one thread per element loads `gate` and `up`, computes
SiLU with one exponential, multiplies, and writes once. The readable version
expresses the same thing as separate absolute-value, exponential, division,
selection, and multiplication steps over whole tensors. The kernel writes no
intermediate tensors and runs in one launch. The measurement at the end tells
you whether that matters for the whole model.

Use it in the MLP. The `swiglu` checkpoint keeps your RMSNorm and RoPE:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k task_3
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-0.6b --max-tokens 16
```

The test compares your output with the readable gate/up equation.

## Task 4: Verify the Cumulative Model

Task 4 adds no new kernel. It checks that the three kernels are wired in at the
right checkpoints. Look over the switches in `Qwen3ModelWeek2.__init__` and the
call sites in `Qwen3MultiHeadAttention.__call__` and `Qwen3MLP.__call__`:
`rmsnorm` should turn on only your norm, `rope` should add your RoPE, and
`swiglu` should add your SwiGLU. The checkpoint features are listed in
`WEEK2_CHECKPOINT_FEATURES`.

The supplied test runs a 10-token input through `simd-matmul`, `rmsnorm`,
`rope`, and `swiglu` on two tiny models, each run with a fresh cache, and
checks that each checkpoint's output stays close to the one before it. It
compares outputs, so it cannot tell which kernel ran; reading the wiring is
your check for that.

Run the whole day:

```bash
pdm run build-ext
pdm run build-ext-ref
pdm run test-refsol --week 2 --day 4
pdm run test --week 2 --day 4
```

Keep `qwen3_week1.py` on its Week 1 operators, and keep the Week 2 interfaces
usable by the Week 3 model. The model's default checkpoint is `selected`, the
finished Week 2 model, so always name the checkpoint you mean when you compare
results.

If you write your own operator benchmark, compare the same logical operation.
Your RoPE takes `B, L, H, D`, but `mx.fast.rope` expects `B, H, L, D`, so
transpose before calling MLX and back afterwards. Without that, a one-token
benchmark treats the heads as sequence positions and times a different
operation. The supplied operator benchmark already does this. Remember to call
`mx.eval` inside every timed iteration.

## Measure the Cumulative Product

Once the full Day 4 gate passes, run the Day 3 control and all three Day 4
checkpoints together with the same cached 0.6B model and workload. Keeping the
`simd-matmul` row in the same run means a gain from one operator cannot hide a
loss from another:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-simd-matmul \
  --variant week2-rmsnorm --variant week2-rope --variant week2-swiglu \
  --variant mlx --model qwen3-0.6b \
  --input-len 128 --output-len 129 --warmup 2 --prefill-logits last \
  --json-output week2-day4-product.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case simd-matmul:decode:128 --case swiglu:decode:128 \
  --case simd-matmul:prefill:128 --case swiglu:prefill:128 \
  --warmup 4 --iterations 12 \
  --json-output week2-day4-attribution.json
```

First check that this run's `simd-matmul` row is close to the control you
saved. The profiler replays the same decode and prefill steps for `simd-matmul`
and `swiglu`, with the same 128-token context, cache bound, and logit setting.
The normalization, position, and activation group is where Day 4's kernels
show up; the other groups replay the same operators in both cases. Report each
checkpoint's whole-model change and the replayed group change separately: the
replay times your operators, not your model, so a faster group alone does not
mean a faster request, and it does not show which kernel your model called.
Write down which group now takes the most replayed time, and what result would
make you undo one of these kernels. The
[performance appendix](./appendix-performance.md) keeps older M4 Pro and 4B
results as history, not as claims about your checkout. A 4B comparison is
optional: cache the model first and rerun every row at 4B.

If you need to move on without one of these kernels, keep its Python interface
and use the matching MLX operator or the readable equation behind it. The rest
of your model still runs your code. That is different from `--solution mlx`,
which runs a separate complete model.

Next is [Day 5: Tiled Dense Prefill Attention](./week2-05-tiled-prefill-attention.md),
with this `swiglu` measurement as its control.

{{#include copyright.md}}
