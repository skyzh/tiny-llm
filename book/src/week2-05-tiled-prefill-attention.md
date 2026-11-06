# 🚧 Week 2 Day 5: Tiled Dense Prefill Attention

After [Day 4](./week2-04-fused-model-kernels.md), your `swiglu` checkpoint runs
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
  --prefill-logits last --json-output week2-day5-control.json
```

The `mlx` row is a separate complete model, there for scale. Build the
reference extension and run its Day 5 tests to see the finished behavior;
your own extension still has its TODOs:

```bash
pdm run build-ext-ref
pdm run test-refsol --week 2 --day 5
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
all. For each source tile:

1. load a 32 × 128 Q tile and a 16 × 128 K tile into
   threadgroup memory;
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
pdm run test --week 2 --day 5 -- -k task_1
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
pdm run test --week 2 --day 5 -- -k task_2
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
The `tiled-prefill` checkpoint keeps the bounded cache, packed projections,
and all three Day 4 kernels; only supported prefill attention changes:

```bash
pdm run build-ext
pdm run test --week 2 --day 5 -- -k task_3
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint tiled-prefill --model qwen3-0.6b --max-tokens 16
```

This test runs `tiled-prefill` and `swiglu` on a tiny model with a 3-token
input, which must take the readable path, and a 10-token input, which
qualifies for the tiled kernel, and compares the outputs. Matching outputs do
not show which kernel ran. The 128-token benchmark below uses the tiled path
in the real model.

## Finish the Cumulative `selected` Model

`selected` has the same features as `tiled-prefill`: request-bounded KV
capacity, packed W4 projections, SIMD matrix prefill, register-cached
RMSNorm, RoPE, SwiGLU, and tiled prefill attention, with the short-query and
wide-row fallbacks. Three of these can be switched off one at a time through
the constructor (`use_bounded_kv_capacity`, `use_register_cached_rms_norm`,
`use_tiled_prefill_attention`), so you can measure what each one is worth.
`selected` is also the model's default checkpoint.

```bash
pdm run test --week 2 --day 5 -- -k selected
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint selected --model qwen3-0.6b --max-tokens 16
```

The test checks that `selected` is the default, and that switching off each of
the three controls still gives close outputs. Then run the whole day:

```bash
pdm run test --week 2 --day 5
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

## Measure the Matched Product

First compare a new `swiglu` row with the one you saved before editing. Then
compare `swiglu`, `tiled-prefill`, `selected`, and MLX with the same cached
0.6B request, machine, logit mode, and warmups:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-swiglu --variant week2-tiled-prefill \
  --variant week2-selected --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day5-product.json
```

A 128-token prompt is above the nine-token threshold, so prefill uses your
kernel. Report prefill and whole-request results separately. With
`--solution tiny_llm`, `tiled-prefill` and `selected` turn on the same
features, so do not count a difference between them as a second
improvement. With
`--solution tiny_llm_ref`, `selected` also includes the reference scheduling
above; report those rows separately and do not credit the difference to tiled
attention.

To see what dominates prefill now:

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case swiglu:prefill:128 --case tiled-prefill:prefill:128 \
  --case selected:prefill:128 --warmup 4 --iterations 12 \
  --json-output week2-day5-attribution.json
```

The profile replays groups of kernels on synthetic token IDs; it does not
replace the whole-request comparison. Note whether the tiled path ran and what
takes the most time now. An operator result, or an older 4B number, cannot
show a speedup for your 0.6B checkout. The
[performance appendix](./appendix-performance.md) keeps earlier measurements,
including rows that could not be run, as history. A 4B comparison is
optional: cache the model and rerun every row at 4B.

That completes Week 2: a tested model that serves one request at a time.
[Week 3](./week3-overview.md) starts from it and adds what serving many
requests needs: batching, chunked prefill, and paged K/V storage.

{{#include copyright.md}}
