# 🚧 Week 2 Day 7: Fused Packed-W4 Gate+Up and SwiGLU

Day 5 gives matrix-shaped packed W4 projections a cooperative schedule. The
component sweep now exposes a more precise MLP question: the gate and up
projections read the same activation, use the same packed-W4 group-of-128
layout, and feed SwiGLU immediately. Can one bounded operator share that input
work and remove an intermediate dispatch without changing the down projection?

This replaces the old Split-K experiment. Split-K helped one isolated
32-token projection by about 5%, but the fixed 128-token product controls did
not improve. That result does not justify making Split-K the final Week 2 path.

Day 7 is independent of optional Day 6. Begin from the Day 5 checkpoint and
finish with a matched keep-or-reject decision.

## Task 1: Freeze the MLP Contract

The learner-owned seam is:

```python
quantized_gate_up_swiglu(x, w_gate, w_up)
```

An equivalent public organization is valid. It must compute the two packed W4
projections from one activation and return their SwiGLU combination:

```plain
gate = dequantize(w_gate) @ x
up = dequantize(w_up) @ x
hidden = silu(gate) * up
```

Preserve Day 5's packed layout, group size 128, transpose convention, scale and
bias behavior, BF16 storage and output, FP32 accumulation, and partial-tile
masks. Keep the down projection unchanged.

Use rows 1, 32, 128, 512, and 2,048 for operator evidence. The product safety
controls include decode and an 8K prompt so a local MLP improvement cannot hide
a larger request regression.

The canonical selector is `fused-gate-up`. Check the implementation, then run
the targeted row sweep and the Day 5/candidate product matrix:

```bash
pdm run build-ext
pdm run test --week 2 --day 7

pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section fused-gate-up \
  --context 1 --context 32 --context 128 --context 512 --context 2048 \
  --context-repeats 2 --warmup 12 --iterations 60 \
  --json-output week2-day7-operator.json

pdm run bench-week2-progression --offline --solution tiny_llm --matrix \
  --variant week2-simd-matmul --variant week2-fused-gate-up \
  --prompt-length 128 --prompt-length 512 \
  --prompt-length 2048 --prompt-length 8192 --prompt-length 32640 \
  --output-len 128 --warmup 2 --repeats 4 --prefill-logits last \
  --json-output week2-day7-matrix.json

pdm run bench-week2-progression --offline --solution tiny_llm --matrix \
  --variant week2-simd-matmul --variant week2-fused-gate-up \
  --prompt-length 128 --prompt-length 512 \
  --prompt-length 2048 --prompt-length 8192 --prompt-length 32640 \
  --output-len 128 --warmup 2 --repeats 4 --prefill-logits last \
  --disable-week2-fused-gate-up \
  --json-output week2-day7-disabled.json
```

## Task 2: Fuse Only the Shared-Input Work

Build one bounded candidate that loads the activation tiles once, evaluates
gate and up against their respective packed weights, and applies SwiGLU before
returning the MLP hidden tensor. Fill `supports_fused_gate_up` and
`quantized_gate_up_swiglu`, then connect them in `Qwen3MLP.__call__`. In the
extension, complete `tiny_llm_ext::quantized_gate_up_swiglu`,
`Week2QuantizedGateUpSwiGLU::eval_gpu`, and the
`week2_quantized_gate_up_swiglu` Metal kernel. Do not fold the down projection
into this operator: it consumes the SwiGLU result and has a different
dependency.

The dispatcher must:

- select the candidate only for the declared packed-W4/BF16 shapes;
- count selections so the matched run proves the candidate executed;
- expose a disable-only control that restores the Day 5 gate, up, and SwiGLU
  path;
- fall back safely for unsupported dtype, layout, group size, transpose,
  dimension, or tail;
- preserve public model outputs and the Day 5 short-row/matrix dispatch rules.

Test row 1, the matrix rows, partial output tiles, invalid layouts, and the exact
fallback. Equivalent tiling and helper names are allowed; the public result and
controls are not.

## Task 3: Attribute the Targeted Phase

Run the Day 5 control and `fused-gate-up` in balanced order for rows 32, 128,
512, and 2,048. Record the gate+up+SwiGLU time, total targeted-phase time,
selection count, and disable-control result. Keep row 1 as the decode-shaped
correctness and safety point.

The normalized isolated evidence that selected this experiment is not itself a
pass: MLP projections plus SwiGLU account for 63.14% of the 128-token prefill
replay and 32.74% at 32,640 tokens. Those shares say where a candidate might
matter. Only the matched candidate comparison says whether this one helps.

## Task 4: Close the Product Gate

Generate exactly 128 output tokens and compare the full Day 5 and Day 7 model
at the fixed prompt matrix. Token 1 remains part of prefill/TTFT, and TPOT uses
the remaining 127 decode intervals. Include the 8K prompt and the disable-only
control.

Keep the candidate only if all of these conditions hold:

1. targeted-phase time improves by at least 5% at either 512 or 2K rows;
2. it moves in the same favorable direction in at least three of the four
   matched 32/128/512/2K pairs;
3. neither the 8K product control nor decode regresses by more than 2%;
4. disabling `fused-gate-up` removes the measured gain.

Otherwise record `reject` or `inconclusive` and retain the Day 5 path. Do not
turn a correct kernel or an isolated share into a product speedup claim.

Finish with the week's decision ledger:

| Step | Evidence that selected it | Same-workload result | Decision and falsifier |
|---|---|---|---|
| KV cache | Full-prefix recomputation | Matched Week 1 versus cache | Your observation |
| Packed W4 | Cached decode attribution | Repeated decode product and attribution | Your observation |
| Fused pointwise | Post-W4 re-profile | Repeated decode product and attribution | Your observation |
| SIMD prefill | Matrix-shaped packed projections | Repeated prefill product and attribution | Your observation |
| Long-context attention | Attention share grows with context | 128/512/2K/8K matrix plus disable control | `keep`, `reject`, `inconclusive`, or skipped |
| Fused gate+up | Shared-input MLP component cost | 32/128/512/2K phase pairs plus 8K/decode controls | `keep`, `reject`, or `inconclusive` |

Close the week with the causal story: what dominated at short and long context,
what changed, what the identical remeasurement showed, and what you chose not
to claim.

{{#include copyright.md}}
