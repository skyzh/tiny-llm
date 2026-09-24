# 🚧 Week 2 Day 3: SIMD Matrix Prefill

Complete [Day 2: Keep W4 Packed](./week2-02-quantize-model.md) first. Its
`quantized-matvec` checkpoint keeps projection weights packed, uses a SIMD
matvec for at most eight activation rows, and falls back to a readable Metal
matrix kernel for larger inputs. Day 3 keeps the same W4 weights and
`quantized_linear` interface. You will make the matrix-shaped prefill path
cooperate across SIMD groups, then compare it with the unchanged Day 2 control.

The supplied partial-tile test makes the work concrete. It multiplies ten
activation rows of width 256 by 97 packed weight rows and expects a 10×97
output. A 32×32 output-tile grid needs one row tile and four column tiles; the
last column tile contains only one valid output. A kernel that handles only
Qwen's aligned main shapes can write outside that output or return wrong edge
values. The test compares your SIMD result with Day 2's readable matrix
control at this shape.

## Keep the pre-edit control

Build both native extensions after completing Day 2. The reference Day 3 test
exercises the supplied implementation independently; the learner's Day 3 test
should reach your unfinished SIMD seam until you implement it:

```bash
pdm run build-ext
pdm run build-ext-ref
pdm run test-refsol --week 2 --day 3
pdm run test --week 2 --day 3 -- -k partial_tiles
```

Before editing the kernel, measure the existing `quantized-matvec` model with
the same cached 0.6B checkpoint used in Day 2. The `--offline` progression
command requires the model files to be cached already. Save the control JSON;
`week2-simd-matmul` is the candidate you will add later:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-quantized-matvec --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day3-control.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case quantized-matvec:prefill:128 --warmup 4 --iterations 12 \
  --json-output week2-day3-control-attribution.json
```

The progression runner uses fresh processes and balanced comparison order.
The profiler attributes one prefill shape. Keep the model, phase, prompt and
output lengths, last-logit mode, warmups, and device fixed for the later
progression run; keep the profiler's case and sampling settings fixed for its
later run. Do not use a prior 4B result as the 0.6B control.

## Task 1: Load and multiply a W4 tile

Work in these learner-owned extension surfaces:

```plain
src/extensions/src/cooperative_matrix.h
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

The operation is still the Day 2 matrix product:

$$
C = A W^T
$$

Read this as: output matrix `C` equals activation matrix `A` multiplied by the
transpose of reconstructed weight matrix `W`. `A` is BF16. `W` is reconstructed
from packed four-bit codes and the stored BF16 scale and bias for each group
of 128 values. The new schedule must preserve Day 2's result, dtype, and
shape; it must not expand the full weight matrix in memory.

Build each 32×32 output tile from 8×8 `simdgroup_matrix` fragments. Four SIMD
groups can each own a 16×16 quadrant of that output tile:

<table>
  <thead>
    <tr><th scope="col">Output rows within a tile</th><th scope="col">Columns 0–15</th><th scope="col">Columns 16–31</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">0–15</th><td>SIMD group 0</td><td>SIMD group 1</td></tr>
    <tr><th scope="row">16–31</th><td>SIMD group 2</td><td>SIMD group 3</td></tr>
  </tbody>
</table>

Each group accumulates its four 8×8 output fragments across reduction slices.
On the 10×97 witness, only the first ten rows and the valid columns in each
output tile may be stored.

For each 32-value slice of the reduction dimension:

1. Load a contiguous BF16 activation tile into threadgroup memory.
2. Unpack the matching W4 codes, apply their stored scale and bias, and place
   the reconstructed BF16 values in a weight tile.
3. Load the 8×8 fragments with the correct row and transposed-weight layout;
   multiply them while accumulating in FP32.
4. Advance to the next reduction slice. Cast once when storing valid output
   elements.

The starter's `CooperativeTileLoader::load` and `CooperativeBlockMMA` helpers
own the full-tile and edge-safe loads, lane-to-fragment mapping, accumulation,
and guarded store. Complete their named TODOs, then implement
`quantized_matmul_simdgroup_w4a16_g128` in
`quantized_matmul.metal`. The transposed fragment view gives the product
`A W^T` without materializing another weight tile. Day 2's Python
`mlx.core` equation and readable Metal matrix kernel remain correctness
controls; the required course path implements its own Metal schedule.

For the 10×97 witness, the activation row tile has ten valid rows, and the
fourth weight-column tile has one valid column. Zero-fill invalid loaded
positions and guard stores so they cannot read or write past these edges. The
supplied partial-tile test checks this output boundary; it does not claim to
cover every reduction length or model shape.

## Task 2: Select the matrix schedule

Keep the Python `quantized_matmul(..., use_simdgroup=True)` seam in
`src/tiny_llm/quantize.py` wired to the native binding. In
`QuantizedMatmul::eval_gpu` in `quantized_matmul.cpp`, keep Day 2's SIMD
matvec when the flattened activation row count `M <= 8`. For `M > 8`, select
your new SIMD matrix kernel when the `use_simdgroup` flag is set; keep the
vanilla matrix kernel callable as a bring-up control. Preserve the existing
rank, shape, dtype, contiguity, group-size, and bit-width validation before
encoding Metal buffers. Match the C++ buffer indices and launch grid to the
Metal entry rather than adding a second public binding.

The `simd-matmul` checkpoint turns on this matrix path in the Week 2 model. It
inherits bounded KV capacity and packed W4 projections from Days 1 and 2.
Thread the model's `use_simdgroup_matmul` choice into the packed projections,
including the tied output head, so the live model uses your kernel for
matrix-shaped work. Fused RMSNorm, RoPE, and SwiGLU are later lessons; no
decode-attention or Split-K branch is part of this checkpoint.

Check the short model setup first, then the SIMD operator and larger-prefill
model behavior:

```bash
pdm run build-ext
pdm run test --week 2 --day 3 -- -k task_1
pdm run test --week 2 --day 3 -- -k test_task_2_simdgroup_matmul_matches_readable_partial_tiles_gpu
pdm run test --week 2 --day 3 -- -k test_task_3_simd_matmul_model_prefill_matches_readable_control_gpu
pdm run test --week 2 --day 3

pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint simd-matmul --model qwen3-0.6b --max-tokens 16
```

The short Task 1 test builds public `simd-matmul` and readable packed
`quantized-matvec` checkpoints from the same tiny Qwen fixture, then runs
the same 1×3-token input through each with separate capacity-3 caches. It
evaluates both outputs, checks BF16 dtype and equal full shapes with a 1×3
leading shape, then compares values with absolute tolerance 0.25 and relative
tolerance 0.01. This checks observable short-input behavior; it does not assert
which internal kernel ran.
The source's `M <= 8` branch retains Day 2's matvec fallback for this input,
so Task 1 can pass before you implement the SIMD matrix kernel.
The direct Task 2 test compares that kernel with the readable control on
partial output tiles. The separate Task 3 test builds tiny Qwen fixtures with
two fixed MLX random seeds, 0 and 4. For each fixture, it runs the same
1×10-token input through both public Week 2 checkpoints with separate
capacity-10 caches. It evaluates both BF16 outputs, checks equal full shapes
and dtype, then compares values with absolute tolerance 0.75 and relative
tolerance 0.05. This checks observable model behavior between `simd-matmul`
and readable packed `quantized-matvec`; it does not prove which internal
kernel ran. Run the complete Day 3 gate and the cached model after both matrix
checks.

If you continue without a custom matrix schedule, keep the course-owned
`quantized_linear` interface and substitute `mx.quantized_matmul` only at this
operator boundary. `--solution mlx` runs a separate complete model and does
not fill your learner cache or model TODOs.

## Measure the matched product

Once the complete Day 3 gate passes, repeat the same 0.6B workload with the
Day 2 control and the new candidate in one progression run:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-quantized-matvec --variant week2-simd-matmul --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day3-product.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case quantized-matvec:prefill:128 --case simd-matmul:prefill:128 \
  --warmup 4 --iterations 12 \
  --json-output week2-day3-attribution.json
```

Check that the post-edit `quantized-matvec` row remains comparable to the saved
pre-edit control. Then report the product prefill change and the attributed
projection change separately. An operator reduction alone does not establish
a whole-model speedup. If you repeat the comparison with the optional 4B
model, cache it first and rerun every compared row at 4B; do not mix model
sizes or prefill-logit modes.

Record what dominated the baseline prefill, whether the candidate changed
that category and the matched product phase, and what result would make you
revise the schedule. The [performance appendix](./appendix-performance.md)
keeps older hardware observations separate from this exercise. This checkout
stops at Day 3; the fused model primitives and tiled attention are future
learner checkpoints, while the older seven-day pages remain
[historical material](./week2-02-benchmark-profile.md).

{{#include copyright.md}}
