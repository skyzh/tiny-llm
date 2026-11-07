# 🚧 Week 2 Day 3: SIMD Matrix Prefill — Stop Re-reading Weights per Row

After [Day 2](./week2-02-quantize-model.md), decode is fast: one activation
row meets each packed weight, and your SIMD matvec handles it. Prefill is a
different shape. A 128-token prompt sends 128 activation rows through every
projection at once, and with more than eight rows, Day 2 routes them to the
simple kernel, where each thread computes one output on its own.

Count what that costs. To compute row `i` of the output, a thread reads all of
weight row `k`. The thread for row `i + 1` reads weight row `k` again, and so
does every other row. With 128 prompt rows, every weight is fetched and
unpacked 128 times, and every activation is fetched once per output column.
Most of that memory traffic is repeats.

Today you fix this with a **tiled** matrix kernel. A group of threads loads one
block of activations and one block of weights into fast on-chip memory, and
every thread in the group reuses them. Each loaded weight then serves 32 rows
instead of one. Apple GPUs have hardware support for small matrix products,
`simdgroup_matrix`, which multiplies 8×8 blocks inside one SIMD group. You will
build the tile out of those.

The weights stay packed, `quantized_linear` keeps its interface, and decode
keeps Day 2's matvec. Only the path for more than eight rows changes, and the
new checkpoint is called `simd-matmul`.

## A Shape That Does Not Fit

The tests start with an awkward shape on purpose: 10 activation rows of width
256, times 97 packed weight rows, giving a 10 × 97 output. Cut into 32 × 32
output tiles, that needs one row of tiles and four columns of tiles:

```plain
output columns:  0 ........ 31 | 32 ....... 63 | 64 ....... 95 | 96
                 tile 0          tile 1          tile 2          tile 3
rows 0-9 valid,  rows 10-31 of every tile are padding
                                                                 only column 96 is real
```

A kernel that works only on Qwen's neatly aligned shapes would either write
past the end of this output or return wrong values at the edges. The test
compares your kernel's result with Day 2's simple kernel at exactly this
shape.

## Keep the Pre-Edit Control

Build both extensions. The reference tests show what a finished Day 3 does;
your own Day 3 test will fail at the SIMD kernel until you write it:

```bash
pdm run build-ext
pdm run build-ext-ref
pdm run test-refsol --week 2 --day 3
pdm run test --week 2 --day 3 -- -k partial_tiles
```

Before you change anything, time the current `quantized-matvec` model and
replay one 128-token prefill with the supplied profiler. The `--offline` run needs the 0.6B model cached
already. Save both files; they are today's control:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-quantized-matvec --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day3-control.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case quantized-matvec:prefill:128 --warmup 4 --iterations 12 \
  --json-output week2-day3-control-attribution.json
```

The profiler case `quantized-matvec:prefill:128` replays one prefill of 128
prompt tokens into an empty cache, through the public operators this
checkpoint selects. With the default `--prefill-logits last`, only the last
row goes through the final norm and output head, matching the benchmark's
serving setup; `all` would put all 128 rows through the head. The cache
bound defaults to `C = S = 128`. This replay times the operators on prepared
inputs; it is not a run of your model's prefill, which the benchmark times.

Read the control replay before you write any code. Today's hypothesis is that
the projections group leads it, because every prompt row re-reads and
re-unpacks every weight in the readable kernel. If that is what you see, the
tiled kernel should shrink that group and raise prefill throughput, while
decode stays where it was. If another group leads, write it down; Days 4 and
5 still need today's kernel, and the end-of-day comparison will show what it
changed.

When you measure again at the end, keep everything the same: model, phase,
prompt and output lengths, logit mode, warmups, and machine for the benchmark,
and the same case, cache bound, logit setting, and iteration counts for the
profile. Do not use an older 4B result, or a result from an earlier version of
the profiler, as the 0.6B control.

## Task 1: Load and multiply a W4 tile

```plain
src/extensions/src/cooperative_matrix.h
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

The math is unchanged from Day 2:

$$
C = A W^T
$$

`A` is the BF16 activation matrix and `W` is the weight matrix rebuilt from
4-bit codes and each group's stored scale and bias. Your new kernel must give
the same result, dtype, and shape as Day 2's, and it must never build the full
`W` in memory.

### How One Tile Is Computed

Each threadgroup computes one 32 × 32 block of `C` with four SIMD groups. Each
SIMD group owns a 16 × 16 quarter of it, which is four 8 × 8 fragments:

<table>
  <thead>
    <tr><th scope="col">Output rows within a tile</th><th scope="col">Columns 0–15</th><th scope="col">Columns 16–31</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">0–15</th><td>SIMD group 0</td><td>SIMD group 1</td></tr>
    <tr><th scope="row">16–31</th><td>SIMD group 2</td><td>SIMD group 3</td></tr>
  </tbody>
</table>

The kernel walks the reduction dimension 32 values at a time. For each slice:

1. Load a 32 × 32 block of BF16 activations into threadgroup memory.
2. Unpack the matching 4-bit weight codes, apply their group's scale and bias,
   and store the rebuilt BF16 weights in a second threadgroup tile.
3. Each SIMD group loads its 8 × 8 fragments from the two tiles and multiplies
   them, adding into FP32 accumulators.
4. Move to the next slice.

When the whole reduction is done, each SIMD group converts its accumulators
to BF16 once and writes them out, skipping any position outside the real
output.

Notice what step 2 buys. One rebuilt weight tile is used by all 32 rows of the
output tile, so each weight is unpacked once per tile instead of once per
output row.

### What to Write

The starter's `cooperative_matrix.h` contains two helpers with
`TODO(Week 2 Day 3)` markers. `CooperativeTileLoader::load` copies a tile
into threadgroup memory, with a fast path for full tiles and a careful path
for edges. `CooperativeBlockMMA` maps each SIMD lane to its fragment elements,
zeroes the accumulators, multiplies fragments, and stores the result with
bounds checks. Fill in those TODOs, then implement
`quantized_matmul_simdgroup_w4a16_g128` in `quantized_matmul.metal` using
them.

The weights are stored as `[output, reduction]`, but the product needs `Wᵀ`.
Read the weight fragments through a transposed view instead of building a
transposed copy.

For the 10 × 97 shape, the activation tile has 10 real rows and the last weight
tile has one real column. Fill every out-of-range position you load with zero,
and guard every store, so the kernel never reads or writes past an edge. The
partial-tile test checks this case; it does not cover every reduction length
or model shape.

Day 2's Python `mlx.core` equation and simple Metal kernel are still your
correctness controls.

## Task 2: Select the matrix schedule

Now route prefill to the new kernel. In `QuantizedMatmul::eval_gpu` in
`quantized_matmul.cpp`:

- keep Day 2's matvec when the flattened row count `M <= 8`;
- for `M > 8`, use `quantized_matmul_simdgroup_w4a16_g128` when the
  `use_simdgroup` flag is set; and
- keep the simple matrix kernel reachable when the flag is not set, as your
  control.

Keep the existing checks on rank, shape, dtype, contiguity, group size, and
bit width before encoding anything. Match the C++ buffer indices and launch
grid to the new Metal kernel; do not add a second public binding. On the
Python side, `quantized_matmul(..., use_simdgroup=True)` in
`src/tiny_llm/quantize.py` should already pass the flag through.

Then turn it on in the model. The `simd-matmul` checkpoint sets
`simdgroup_matmul` in `WEEK2_CHECKPOINT_FEATURES`, on top of Day 1's bounded
cache and Day 2's packed weights. The choice travels in three steps:

1. the model loader passes it to each packed projection as
   `QuantizedWeights.from_mlx_layer(..., use_simdgroup_matmul=...)`,
   including the tied output head;
2. `quantized_linear` reads `w.use_simdgroup_matmul` and, for more than eight
   rows, calls `quantized_matmul(..., use_simdgroup=w.use_simdgroup_matmul)`;
3. the native primitive sees `use_simdgroup` and selects your kernel.

If any step drops the flag, prompts quietly keep using the simple kernel.
Normalization, RoPE, and SwiGLU kernels are Day 4; this checkpoint has no
other new kernels.

```bash
pdm run build-ext
pdm run test --week 2 --day 3 -- -k task_1
pdm run test --week 2 --day 3 -- -k test_task_2_simdgroup_matmul_matches_readable_partial_tiles_gpu
pdm run test --week 2 --day 3 -- -k test_task_3_simd_matmul_model_prefill_matches_readable_control_gpu
pdm run test --week 2 --day 3

pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint simd-matmul --model qwen3-0.6b --max-tokens 16
```

The test file numbers its groups differently from this chapter, so here is
what each command checks:

- **`-k task_1`** runs a 3-token input through the `simd-matmul` and
  `quantized-matvec` checkpoints of a tiny model and compares the outputs.
  Three rows is at most eight, so this still uses Day 2's matvec and can pass
  before your new kernel works. It checks that the checkpoint is wired up.
- **The `task_2` test** compares your new kernel with the simple kernel on the
  10 × 97 partial-tile shape.
- **The `task_3` test** runs a 10-token input through both checkpoints on two
  tiny models (random seeds 0 and 4), so the new kernel is used inside the
  model.

None of them check which kernel actually ran; they compare outputs. Run the
full Day 3 gate and the live model once both kernel checks pass.

If you need to move on without the tiled kernel, keep `quantized_linear` as it
is and use `mx.quantized_matmul` inside it for the matrix case only. That is
different from `--solution mlx`, which runs a separate complete MLX model and
does none of your cache or model work.

## Measure the matched product

Once the full Day 3 gate passes, run the Day 2 control and the new checkpoint
in one progression, and replay both:

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

First check that this run's `quantized-matvec` row is close to the control you
saved before editing; if it is not, something about the machine changed and
the comparison is not fair. Then report two things separately: how much
prefill throughput changed for the whole model, and how much the projections
group changed in the replay. A faster operator in the replay does not by
itself mean a faster model, and the replay does not show that your model
routes prompts to the new kernel; that is what the three-step flag check in
Task 2 and the full-model tests are for.

Decode still runs Day 2's matvec, so expect the change to show up mainly in
prefill. If you
repeat this with the optional 4B model, cache it first and rerun every row at
4B; do not mix model sizes or logit modes.

Write down which group took the most replayed prefill time before the change,
whether the new kernel reduced it, how the whole-model prefill result moved,
and what result would make you rethink the tiling. Then replay decode for the
new checkpoint (`--case simd-matmul:decode:128`) and look at the groups other
than projections. The normalization, position, and activation group is Day 4's
starting point; if it is small on your machine, note that before you go on.
The [performance appendix](./appendix-performance.md) keeps older hardware
results separate from this exercise.

Next, [Day 4: Fused Model Primitives](./week2-04-fused-model-kernels.md) starts
from this `simd-matmul` result as its control, and
[Day 5](./week2-05-tiled-prefill-attention.md) adds tiled prefill attention.
Lessons from the older seven-day Week 2 remain as
[historical material](./week2-02-benchmark-profile.md).

{{#include copyright.md}}
