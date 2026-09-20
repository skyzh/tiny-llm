# 🚧 Compact Primitive Lab: Finish One Operator at a Time

The incoming `simd-matmul` checkpoint now has separate projection schedules for
decode and prefill. Norms, positional rotations, and the elementwise activation
still use the readable Week 1 equations. You will keep those equations as
controls and replace each with a compact Metal operator.

Work in `src/tiny_llm/week2_kernels.py` and the existing
`src/extensions/src/week2_kernels.cpp` and `.metal` shells. The declarations,
bindings, and build registration are supplied. You own each wrapper, native
primitive, kernel body, and model call site.

Begin with RMSNorm only:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k rms
```

Initial failures should identify the unimplemented RMSNorm path. Do not turn on
RoPE or SwiGLU early merely to make a cumulative model call work.

## RMSNorm: Keep the Reduction with Its Output

For one hidden row, RMSNorm uses the mean squared value to normalize every
coordinate, then applies the learned weight:

$$
y_i = x_i\,\operatorname{rsqrt}\left(\frac{1}{D}\sum_j x_j^2+\epsilon\right)w_i.
$$

A single operator can accumulate the squared sum in FP32 and use that result to
write normalized output without materializing a squared-input tensor. Begin
with a simple lane-strided reduction. If you divide a row among several SIMD
groups, their partial sums need a second reduction and synchronization before
any group uses the final value.

Complete `FastRMSNorm`, the `rms_norm` extension primitive, and the
`week2_rms_norm` Metal body. Preserve BF16 input/output and use the supplied
tolerance against the readable equation: a different rounding point need not
produce bit-identical output. Wire `FastRMSNorm` into the model's norms at
`rmsnorm`, while retaining matrix dispatch.

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k rms
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint rmsnorm --model qwen3-0.6b
```

## RoPE: Rotate the Right Position

Complete `FastRoPE`, the `rope` primitive, and `week2_rope`. The model-facing
layout is `B, L, H, D`. An offset identifies the first incoming position, so the
rotation for a token uses `offset + token_position`, not its position inside the
small decode call alone.

Accept a scalar offset or one offset per batch row. Normalize it at the wrapper
boundary, and preserve the requested pairing convention. Computing a pair's
angle once lets you rotate both coordinates; reuse across heads can remove
repeated trigonometry, provided the heads use the same position and frequency.
First match the readable result, including nonzero and per-batch offsets.

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k rope
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint rope --model qwen3-0.6b
```

At `rope`, custom RMSNorm and matrix prefill stay enabled. Elementwise SwiGLU
remains the earlier readable expression.

## SwiGLU: Combine Two Existing Tensors

The final primitive consumes the already-computed gate and up projections:

$$
y = \operatorname{SiLU}(g)\odot u
  = \frac{g}{1+\exp(-g)}\odot u.
$$

Complete `swiglu`, its native primitive, and `week2_swiglu`. One output element
needs one gate value and one up value. Keep the intermediate arithmetic in the
kernel and write the BF16 result once. This exercise combines the elementwise
expression; it does not yet combine the two matrix projections that produced
its inputs.

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k swiglu
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-0.6b
```

## Compare Each Addition with Its Predecessor

Run the full gate once all three operators and their checkpoint switches work:

```bash
pdm run test --week 2 --day 4
```

It checks the equations, offset forms, course-owned wrappers, cumulative
composition, and preservation of the Week 1 readable operators. The next
checkpoints must not reach backward and replace the Week 1 control.

Use one matched product ladder to see each addition separately:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-simd-matmul --variant week2-rmsnorm \
  --variant week2-rope --variant week2-swiglu \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-primitives.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case simd-matmul:decode:128 --case swiglu:decode:128 \
  --warmup 4 --iterations 12 --json-output week2-primitives-attribution.json
```

The predecessor checkpoint is the fallback control for each addition. All four
retain matrix prefill, so a comparison cannot accidentally attribute its gain
to a primitive. If the combined result hides a regression, attribute the
individual `rmsnorm` and `rope` cases at the same shape too.

Your next question concerns the inputs to these operators. Q, K, and V read the
same hidden state; the gate and up projections do too. Continue from `swiglu` to
[Shared-Input Fusion](./week2-06-operator-lab.md), where you will test whether
sharing that work benefits the full request.

{{#include copyright.md}}
