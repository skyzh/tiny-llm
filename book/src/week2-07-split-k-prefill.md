# 🚧 Week 2 Day 7: Prefill-Only Fused Packed-W4 Gate+Up/SwiGLU

Day 5 makes packed-W4 projections efficient for matrix-shaped prefill. Gate
and up now expose one more piece of shared work: they read the same activation,
use the same group-of-128 packed layout, and feed SwiGLU immediately. You will
combine those two projections with SwiGLU for a bounded prefill range while
leaving decode and the down projection alone.

The earlier `fused-gate-up` checkpoint selected its fused primitive for decode
as well as prefill. Its isolated row sweep recorded 84.10–88.91% improvements
from 32 through 2,048 rows, but row 1 lost decisively. Routing decode through
that losing shape made the product policy invalid. The replacement keeps the
measured prefill range and makes row 1 an explicit fallback.

Day 7 begins from Day 5 and is independent of optional Day 6. Its public
checkpoint is `prefill-fused-gate-up`. The operator benchmark keeps the
`fused-gate-up` section name because it measures the primitive rather than a
product dispatch policy.

## Task 1: Fix the Numerical Oracle

The learner-owned primitive is:

```python
quantized_gate_up_swiglu(x, w_gate, w_up)
```

It returns one MLP hidden tensor:

```plain
gate = fp32_projection(x, bf16_dequantize(w_gate))
up = fp32_projection(x, bf16_dequantize(w_up))
hidden = silu(gate) * up
return bf16(hidden)
```

Use BF16-dequantized weights, accumulate both projections and SwiGLU in FP32,
and cast once at the end. Preserve Day 5's packed `uint32` W4 layout, four-bit
codes, group size 128, transpose convention, scales, biases, and partial output
tiles. The down projection consumes the returned hidden tensor separately; it
does not belong in this fusion.

Complete the learner-owned `quantized_gate_up_swiglu` wrapper, the existing
`tiny_llm_ext::quantized_gate_up_swiglu` operation and
`Week2QuantizedGateUpSwiGLU::eval_gpu` body, and the
`week2_quantized_gate_up_swiglu` Metal kernel. Reuse that operation rather than
adding another native entry point.

Run the focused checkpoint after the primitive matches the deterministic
oracle:

```bash
pdm run build-ext
pdm run test --week 2 --day 7
```

The focused feedback covers the boundary rows 1/31/32/128/512/2,048/2,049,
metadata and tail fallback, the row-1 route and counters, disable behavior, and
the unchanged down projection. A passing numerical test does not show that the
product should select the primitive.

## Task 2: Select Prefill, Never Decode

Implement `supports_prefill_fused_gate_up` and connect the policy in
`Qwen3MLP.__call__`. Select the fused primitive only when all of these
conditions hold:

- the selector is enabled;
- the flattened activation has `32 <= M <= 2048` rows and BF16 dtype;
- gate and up have matching packed `uint32` W4 weights;
- both use four-bit values, group size 128, BF16 scales and biases, and
  identical input/output metadata; and
- the output tiles, including a partial final tile, are valid.

Use separate gate, up, and SwiGLU for row 1 decode, rows 2–31, rows above 2K,
unsupported metadata or layouts, invalid tails, and the disable control. Keep
fused and separate counters independent. Disabling the checkpoint must restore
the complete Day 5 route for every shape.

This policy makes no decode, 8K-or-larger prefill, down-projection, native-edge,
other-model, or fleet claim.

## Task 3: Measure the Operator Boundary

Compare the fused primitive with Day 5's two `quantized_linear` projections
plus `swiglu`. Use the same packed weights, inputs, synchronization, and
execution order:

```bash
pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section fused-gate-up \
  --context 32 --context 128 --context 512 --context 2048 \
  --context-repeats 2 --warmup 12 --iterations 60 \
  --json-output week2-day7-operator.json
```

The primitive passes its operator gate only if it improves by at least 5% at
512 or 2K rows and moves in the favorable direction in at least three of the
four 32/128/512/2K pairs. Keep row 1 in the correctness route as the decode
safety point; do not average it into the prefill claim.

## Task 4: Close the Product Gate

Generate exactly 128 output tokens and compare the full Day 5 and Day 7 models
in balanced fresh processes. The 8K prompt must use separate gate/up/SwiGLU,
which makes it a fallback and regression control rather than an eligible win:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --matrix \
  --variant week2-simd-matmul \
  --variant week2-prefill-fused-gate-up \
  --prompt-length 128 --prompt-length 512 \
  --prompt-length 2048 --prompt-length 8192 \
  --output-len 128 --warmup 2 --repeats 4 --prefill-logits last \
  --json-output week2-day7-matrix.json

pdm run bench-week2-progression --offline --solution tiny_llm --matrix \
  --variant week2-simd-matmul \
  --variant week2-prefill-fused-gate-up \
  --prompt-length 128 --prompt-length 512 \
  --prompt-length 2048 --prompt-length 8192 \
  --output-len 128 --warmup 2 --repeats 4 --prefill-logits last \
  --disable-week2-prefill-fused-gate-up \
  --json-output week2-day7-disabled.json
```

Token 1 belongs to TTFT, and TPOT covers the remaining 127 decode intervals.
Apply the complete fixed gate:

1. the operator gate above passes;
2. median TTFT improves by at least 5% at 512 or 2K;
3. fused dispatch is zero for every row-1 decode;
4. no median TPOT at 128/512/2K/8K regresses by more than 2%;
5. 8K prefill uses the separate fallback and its median TTFT does not regress
   by more than 2%; and
6. disabling the selector removes the eligible-prefill gain.

Keep the policy only if every condition holds. Otherwise record `reject` or
`inconclusive` and retain Day 5's separate route. Performance for the
integrated checkpoint is pending until this exact operator and product matrix
is measured.

Finish the week with a causal ledger:

| Step | Evidence that selected it | Same-workload result | Decision and falsifier |
|---|---|---|---|
| KV cache | Full-prefix recomputation | Matched Week 1 versus cache | Your observation |
| Packed W4 | Cached decode attribution | Repeated decode product and attribution | Your observation |
| Fused pointwise | Post-W4 re-profile | Repeated decode product and attribution | Your observation |
| SIMD prefill | Matrix-shaped packed projections | Repeated prefill product and attribution | Your observation |
| Context-selected attention | Attention share grows with context | 128/512/2K/8K matrix plus dispatch and disable controls | `keep`, `reject`, `inconclusive`, or skipped |
| Prefill-fused gate+up | Shared-input MLP cost and losing row 1 | 32/128/512/2K operator pairs plus product controls | `keep`, `reject`, or `inconclusive` |

Close with what dominated at short and long context, what the matched reruns
showed, and which shapes you deliberately left on the readable route.

{{#include copyright.md}}
