# 🚧 Week 2 Day 4: Fused Model Primitives

Complete [Day 3: SIMD Matrix Prefill](./week2-03-simd-matrix-prefill.md)
first. You can now run a cached model whose packed W4 projections use a SIMD
matrix schedule for prompt rows. RMSNorm, RoPE, and SwiGLU still follow the
readable Week 1 equations around those projections. Day 4 replaces each of
these three operators in turn while keeping the same model, cache, and
`quantized_linear` path.

Before changing an operator, run the cached 0.6B `simd-matmul` checkpoint and
save a matched product control. The `--offline` runner needs the model files
cached already; use the Day 3 model if you have it. The `mlx` row is a separate
full-model baseline, not a replacement for your Week 2 model:

```bash
pdm run build-ext
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-simd-matmul --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day4-control.json
```

The learner-owned extension shells are already present:

```plain
src/tiny_llm/week2_kernels.py
src/extensions/src/week2_kernels.cpp
src/extensions/src/week2_kernels.metal
```

Build the reference extension and run its Day 4 checks as a completed
control. The focused learner check copies the supplied test into `tests/` and
should stop at the named `FastRMSNorm` TODO before you implement Task 1:

```bash
pdm run build-ext-ref
pdm run test-refsol --week 2 --day 4
pdm run test --week 2 --day 4 -- -k task_1
```

Implement and integrate RMSNorm, then RoPE, then SwiGLU. Each checkpoint
includes all earlier Day 4 changes: `rmsnorm`, `rope`, and `swiglu`. After
each operator, first check its numerical result and then run the live model
at that checkpoint. A passing operator comparison alone cannot show that the
model uses it.

RMSNorm, RoPE, and SwiGLU surround the projections in every transformer layer.
Week 1 gives you readable Python `mlx.core` equations; your Week 2 path keeps
their interfaces and supplies purpose-built Metal kernels.

Your solution still uses MLX arrays and its extension API. MLX schedules the
graph node, owns its buffers, and dispatches the Metal function; the course
path owns the arithmetic inside that function. The Week 1 Python equations
remain readable controls; MLX's optimized operators provide numerical
baselines where applicable.

## Why Fusion Helps

Week 1's Python `mlx.core` equations already become native GPU work inside the
lazy graph. Here, the useful question is how many operations, launches, and
memory passes that graph still describes.

For example, RMSNorm expressed as `mlx.core` operations casts, squares,
reduces, takes a reciprocal square root, multiplies, casts again, and applies a
learned weight. A compiler may fuse some adjacent element-by-element work, but
the row reduction is a boundary. Intermediate values and multiple dispatches
remain possible.

A single Metal kernel gives you explicit control over the whole operator:

- one dispatch contains the reduction or pointwise steps;
- values stay in registers or SIMD-group storage between steps;
- float accumulation is used where numerical stability needs it;
- inputs are read once when practical, and only the final tensor is written;
- the grid matches decode shapes instead of a generic tensor operation.

The MLX graph may already fuse some of its work, so count and time the actual
result on the same request. The source language is not the point; the work the
device performs is.

## Task 1: RMSNorm

Start with the fail-closed `tiny_llm_ext::rms_norm` binding and
`Week2RMSNorm::eval_gpu` in `src/extensions/src/week2_kernels.cpp`. Add Metal
`week2_rms_norm_register_cached` and a readable `week2_rms_norm` fallback in
`src/extensions/src/week2_kernels.metal`, then implement
`FastRMSNorm.__call__` in `src/tiny_llm/week2_kernels.py`. The starter already
provides the header, binding, C++/Metal files, and CMake registration. Keep
those interfaces. `Week2RMSNorm::eval_cpu` in
`src/extensions/src/week2_kernels.cpp` reports the GPU-only error.

The readable equation normalizes each row using its mean squared value, then
applies a learned weight. For a 2,560-element hidden row, one 32-lane SIMD
group would leave roughly 80 serial elements per lane. A fixed-width control
uses 256 threads, or eight groups, per row: each group reduces its portion
with `simd_sum`, lane zero writes one partial sum to threadgroup memory, and
the first group combines those eight sums:

```plain
group_sum = simd_sum(each lane's local sum of squares)
row_sum = simd_sum(the active groups' partial sums)
inverse_rms = rsqrt(row_sum / hidden_size + epsilon)
output[i] = input[i] * inverse_rms * weight[i]
```

For the register-cached path, let each thread retain up to four input values
across the reduction and write them without rereading the row. Round the
required thread count up to a full SIMD group; a 2,560-element row needs 640
threads for four values per thread. Use this path through width 4096 and keep
the fixed-width path as the wider-row fallback. The wrapper can record
`register_cached` and `fixed_width_fallback` dispatch counts as diagnostics.
The primitive validates shape and dtype, allocates the MLX output, binds its
buffers and constants, and provides one float partial sum per active SIMD
group. Instantiate the BF16 kernel. Accumulate the reduction and scale in
float, then cast the output once. The readable Python path may round at a
different intermediate step, so compare with a numerical tolerance rather
than bit equality.

Wire `FastRMSNorm` into every Week 2 norm as soon as the kernel works,
including Q/K, layer, and final norms. Then run the focused check and a live
`rmsnorm` model before touching RoPE:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k task_1
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint rmsnorm --model qwen3-0.6b --max-tokens 16
```

The supplied Task 1 operator check compares BF16 output with an MLX RMSNorm
control at absolute and relative tolerances of 0.02. The separate
register-cache boundary checks include the wider-row fallback.

## Task 2: RoPE

Next complete `tiny_llm_ext::rope` and `Week2RoPE::eval_gpu` in
`src/extensions/src/week2_kernels.cpp`, add the
`week2_rope` function in `src/extensions/src/week2_kernels.metal`, and finish
`FastRoPE.__call__` in `src/tiny_llm/week2_kernels.py`.
`Week2RoPE::eval_cpu` in `src/extensions/src/week2_kernels.cpp`
reports the GPU-only error.

Implement RoPE for the model's native `B, L, H, D` layout. A naive element
kernel calculates the same angle, sine, and cosine separately for both members
of every pair and again for every head. Instead, assign one thread a pair index
and a block of four heads. Compute the angle once, then rotate both elements of
that pair across the four heads:

```plain
angle = (batch_offset + token_position) * base ** (-pair / (dims / 2))
real' = real * cos(angle) - imag * sin(angle)
imag' = imag * cos(angle) + real * sin(angle)
```

Accept either one scalar offset or one offset per batch row in the Python
wrapper. Normalize both cases to an int32 array before dispatch. Supporting
per-batch offsets matters once requests at different decode positions share a
batch.

Unlike a graph that builds position arrays, gathers sine and cosine values,
splits the head, performs several element-by-element operations, and
concatenates the result, this kernel reads each input pair and writes each
rotated element directly. Reusing trigonometry across four heads is the key
optimization. Use Metal's `fast::exp2`, `fast::sin`, and `fast::cos` for the
BF16 path. Normalize a batch's offsets once in the model call,
outside the layer loop, instead of rebuilding the same array in every layer.

Replace the Python `mlx.core` RoPE in the model you have already optimized.
The `rope` checkpoint keeps register-cached RMSNorm active. Test the operator
and run that cumulative model before moving to SwiGLU:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k task_2
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint rope --model qwen3-0.6b --max-tokens 16
```

The supplied Task 2 control checks BF16 RoPE output with per-batch offsets
at absolute and relative tolerances of 0.02.

## Task 3: SwiGLU

Finish the operator sequence with `tiny_llm_ext::swiglu` and
`Week2SwiGLU::eval_gpu` in
`src/extensions/src/week2_kernels.cpp`. Add the `week2_swiglu` function in
`src/extensions/src/week2_kernels.metal`, and `swiglu` in
`src/tiny_llm/week2_kernels.py`.
`Week2SwiGLU::eval_cpu` in `src/extensions/src/week2_kernels.cpp`
reports the GPU-only error.

SwiGLU combines the gate and up branches:

```plain
output = (gate / (1 + exp(-gate))) * up
```

Implement it as one thread per element. That thread loads `gate` and `up`,
evaluates SiLU with one exponential, multiplies the branches, and performs one
output write. The Week 1 form is easier to inspect, but it describes `abs`,
`exp`, division, selection, and multiplication as separate array operations.
The custom kernel writes no intermediate tensor and contains the expression
in one dispatch; the matched measurement will show whether that helps.

Wire the fused expression into the model. The `swiglu` checkpoint retains the
RMSNorm and RoPE changes. Check the operator and run that model:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k task_3
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-0.6b --max-tokens 16
```

The supplied Task 3 control checks BF16 SwiGLU output against the readable
gate/up equation at absolute and relative tolerances of 0.02.

## Task 4: Verify the Cumulative Model

Now verify the cumulative switches in `Qwen3ModelWeek2.__init__` and the call
sites in `Qwen3MultiHeadAttention.__call__` and `Qwen3MLP.__call__`. Task 4 is
composition work: it uses the three functions from Tasks 1–3 and adds no new
extension function. Inspect the model wiring as well as its public outputs;
output agreement alone cannot establish which kernel handled a call.

The supplied Task 4 check uses two fixed-seed tiny model fixtures. For each
fixture it feeds one row of ten tokens into `simd-matmul`, `rmsnorm`, `rope`,
and `swiglu` with a fresh capacity-10 cache for each run. It checks the BF16
output shape and compares each successive pair at absolute tolerance 0.5
and relative tolerance 0.02. Inspect the checkpoint switches and call sites
as a separate wiring check; the numerical test observes public output, not
which Metal function ran.

Once all three kernels are exposed through C++ MLX primitives, run the complete
test file. Keep `qwen3_week1.py` on its Week 1 Python operators, and leave the
Week 2 interfaces reusable by the Week 3 serving model. After Day 4,
`swiglu` is the default Week 2 checkpoint; use explicit checkpoint names in
comparisons so each cumulative step remains visible.

```bash
pdm run build-ext
pdm run build-ext-ref
pdm run test-refsol --week 2 --day 4
pdm run test --week 2 --day 4
```

Use tolerance-based comparisons with the Python reference equations rather
than bit-for-bit equality. Cover both scalar and per-batch RoPE offsets. When
timing these lazy operations, call `mx.eval` inside every measured iteration.

The operator benchmark must also compare the same logical RoPE layout. Your
RoPE kernel accepts the model-native `B, L, H, D` tensor. `mx.fast.rope`
expects `B, H, L, D`, so transpose into that layout before the MLX call and
transpose its result back afterward. Without those transposes, a one-token
benchmark accidentally treats the head axis as sequence positions and the
timing no longer measures an equivalent operation.

## Measure the Cumulative Product

After the complete Day 4 gate passes, measure the Day 3 control and each Day 4
checkpoint together with the same cached 0.6B model, workload, last-logit
mode, device, and warmups. The progression runner balances order and uses
fresh processes. Keep the baseline row so one operator's gain cannot hide a
later regression:

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

Compare the post-edit `simd-matmul` row with the saved pre-edit control first.
Then report each cumulative product change and the attributed operator
category separately. An operator reduction does not establish a complete
request speedup. Record which category now dominates and what result would
make you revise the fusion choice. The
[performance appendix](./appendix-performance.md) keeps older M4 Pro and 4B
observations as historical evidence rather than claims for this checkout. A
4B follow-up is optional: cache that model first and repeat every compared row
at 4B rather than mixing model sizes.

If you continue without writing one of these kernels, keep its public course
interface and substitute the equivalent MLX operator or equation only at that
boundary. The cached Week 2 model and other course-owned operators still run;
`--solution mlx` runs a separate complete model. Day 5's tiled dense prefill
attention remains a future learner checkpoint in this checkout.

{{#include copyright.md}}
