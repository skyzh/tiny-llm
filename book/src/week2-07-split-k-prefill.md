# Historical Week 2: Conditional Split-K

> **Earlier lesson address.** This page preserves the conditional Split-K
> explanation and its original links. The current Week 2 learner route
> ships [Day 1: Cache and Measure](./week2-01-kv-cache.md),
> [Day 2: Keep W4 Packed](./week2-02-quantize-model.md),
> [Day 3: SIMD Matrix Prefill](./week2-03-simd-matrix-prefill.md), and
> [Day 4: Fused Model Primitives](./week2-04-fused-model-kernels.md).
> Later checkpoints, day numbers, tests, and commands below belong to an
> earlier all-days course state; do not use them as gates for this checkout.


Day 5 leaves a reusable 32×32×32 SIMD-matrix projection and an exact unsplit
fallback. Day 6 is an optional branch and is not inherited here: the `split-k`
checkpoint contains the Day 5 SIMD path plus Split-K, without decode attention.

Begin with an under-filled short shape, then return to the fixed 128×129
product workload. Keep Split-K only for shapes where the same-workload evidence
supports it.

## Why Split the Reduction Dimension?

For

$$
C = A W^T,
$$

the Day 5 grid spreads work across output rows and columns. When `M` is small
and the Qwen projection width is narrow, it may launch too few independent
threadgroups to fill the GPU. Split-K creates parallel work along the reduction
dimension:

```plain
for each split s:
    partial[s] = A[:, k_start(s):k_end(s)] @ W[:, k_start(s):k_end(s)].T

C = sum(partial, axis=split)
```

Each split must align to the W4 group size, write to a disjoint partial plane,
and accumulate its local dot product in FP32. A second kernel reduces the
partial planes in FP32 and casts the final output to BF16.

That extra parallelism also adds a dispatch, a temporary buffer, and another
memory pass. Split-K is therefore a shape-conditioned schedule, not an
automatic upgrade.

## Task 1: Freeze a Short-Shape Control

First verify the inherited Day 5 path and record a 32-token attribution pair:

```bash
pdm run build-ext
pdm run test --week 2 --day 7

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case simd-matmul:prefill:32 --case split-k:prefill:32 \
  --warmup 4 --iterations 12 \
  --json-output week2-day7-short-attribution.json
```

Capture the exact source, model, phase, token count, prompt rule, software, and
device. Do not substitute a 128-token baseline for the 32-token candidate.

## Task 2: Reuse the Day 5 Tile for Each Partition

Extend the existing quantized-matmul primitive rather than adding a parallel
public operator. Reuse Day 5's loader, W4 dequantization, and matrix fragments
inside each aligned K partition. Validate that:

- every split begins and ends on a group-of-128 boundary;
- partial planes are disjoint and cover the full reduction exactly once;
- edge rows and columns are masked before load or store;
- accumulation and reduction remain FP32;
- `split_k <= 1` dispatches exactly to the Day 5 unsplit kernel.

The tests grade public results, dtype and shape, valid partitioning, and the
exact fallback. They do not require a private helper name, Metal symbol, or a
particular split-count formula.

## Task 3: Make Dispatch Explicit

Expose the `split-k` checkpoint with an immutable feature set: packed W4,
fused pointwise operators, SIMD prefill, no optional decode-attention branch,
and Split-K only where its policy selects more than one partition.

Keep that public policy in `QuantizedMatmul::eval_gpu`; the supplied starter
surface is `src/extensions/src/quantized_matmul.cpp`. Internal helper and Metal
kernel names remain implementation choices.

Keep the policy small and inspectable. Static dispatch can demonstrate that a
Split-K and reduction kernel exist, but it cannot prove higher occupancy or a
product speedup. Those claims require measured evidence.

If you want to continue without Split-K, preserve `split_k <= 1` and the Day 5
unsplit result. The chapter's learning outcome is the conditional decision,
not an unconditional custom-kernel win.

## Task 4: Re-profile the Short Shape

Rerun the exact 32-token attribution command from Task 1 so the baseline and
candidate differ only in schedule. In the checked M4 Pro example, Split-K
reduced total attributed time by 4.87% and projection time by
5.01%. Its trace exposed only static Split-K and reduction dispatches; no
timeline or counter tree materialized, so no occupancy improvement was
inferred.

Write `keep`, `reject`, or `inconclusive` for the 32-token shape, then name the
result that would reverse your decision. A sub-percent difference is not a
strong conclusion without a larger sample.

## Task 5: Close Week 2 at the Fixed Workload

Return to the Day 5 unsplit checkpoint and compare it with Day 7 at the same
Qwen3-4B 128×129 product control used throughout the week:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-simd-matmul --variant week2-split-k --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day7-final.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case simd-matmul:prefill:128 --case split-k:prefill:128 \
  --warmup 4 --iterations 12 \
  --json-output week2-day7-final-attribution.json
```

On the checked two-sample product control, prefill changed from 721.60 to
718.36 tokens/s (-0.45%) and decode changed by +0.14%. That supports rejecting
Split-K for this fixed 128-token product workload while conditionally retaining
the short-shape experiment. It does not establish a portable crossover.

Finish with the week's decision ledger:

| Step | Evidence that selected it | Same-workload result | Decision and falsifier |
|---|---|---|---|
| KV cache | Full-prefix recomputation | Matched Week 1 versus cache | Your observation |
| Packed W4 | Cached decode attribution | Repeated decode product and attribution | Your observation |
| Fused pointwise | Post-W4 re-profile | Repeated decode product and attribution | Your observation |
| SIMD prefill | Day 4 128-token prefill profile | Repeated prefill product and attribution | Your observation |
| Optional operator lab | Explicit secondary workload | Before/after/fallback record | `keep`, `reject`, `inconclusive`, or skipped |
| Split-K | Under-filled 32-token projection | Short control plus fixed 128×129 control | One decision per shape |

Close the week with the causal story: what dominated, what changed, what the
identical remeasurement showed, and what you chose not to claim.

{{#include copyright.md}}
