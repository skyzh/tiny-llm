# 🚧 Week 2: Make One Request Faster

Your Week 1 model can generate text. Week 2 asks a stricter question: **for one
fixed request, which change removes measured work without changing the model's
answer contract?**

Keep one control request visible all week. After every checkpoint:

1. run the focused diagnostic that exposes the missing seam;
2. complete the supplied correctness gate;
3. run the model at the new checkpoint and its immediate predecessor;
4. record the workload, observation, dispatch or copy counter, and a decision;
5. keep the readable fallback available.

The product command is deliberately simple and accepted by the frozen public
interface. `/usr/bin/time -p` includes process startup and model loading, so it
is a coarse learner observation rather than a kernel benchmark. Use the same
locally available model, prompt, and `--max-tokens` on both sides:

```bash
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b --max-tokens 16
```

Do not compare numbers from different request shapes. A small timing difference
on one run is inconclusive; correctness and path counters come first.

## The Executable Route

The chapter order and public checkpoint order are the same:

| Chapter | Learner-owned seam | Completed checkpoint | First diagnostic |
|---|---|---|---|
| [1. Reuse the prefix](./week2-01-kv-cache.md) | Dense K/V append, model offsets, generation loop | `kv-cache` | `pdm run test --week 2 --day 1 -- -k full_cache` |
| [2. Bound cache movement](./week2-02-benchmark-profile.md) | Logical length, physical capacity, slice writes, rewind/reset | `capacity-cache` | `pdm run test --week 2 --day 2 -- -k logical_prefix` |
| [3. Keep W4 packed](./week2-03-quantize-model.md) | Packed embedding/linear, native W4 matvec | `quantized-matvec` | `pdm run test --week 2 --day 3 -- -k task_1` |
| [4. Reuse matrix tiles](./week2-04-fused-model-kernels.md) | SIMD-group matrix schedule and tail guards | `simd-matmul` | `pdm run test --week 2 --day 4 -- -k partial_tiles` |
| [5. Keep small primitives compact](./week2-05-simd-matrix-prefill.md) | Register-cached RMSNorm, then RoPE and SwiGLU | `rmsnorm` → `rope` → `swiglu` | `pdm run test --week 2 --day 5 -- -k register_cached` |
| [6. Tile dense prefill attention](./week2-06-operator-lab.md) | BQ32/BK16 online softmax, masks, GQA, tails | `tiled-prefill` | `pdm run test --week 2 --day 6 -- -k causal_gqa` |
| [7. Select and explain](./week2-07-split-k-prefill.md) | Independent controls and cumulative decision | `selected` | `pdm run test --week 2 --day 7 -- -k exactly` |

The day numbers above are stable supplied-test selectors. After focused work,
run the whole day's gate and the public checkpoint command shown in its chapter.

![Week 2's executable checkpoint flow and the changing component/product crossover.](./week2-kernel-profile.svg)

## Three Different Kinds of Evidence

- **Correctness and dispatch evidence** comes from the supplied tests and the
  cache/kernel counters. It establishes that the intended path ran.
- **Component evidence** compares one operator with its readable control at the
  same shape. It can explain a mechanism, but it is not request latency.
- **Product evidence** times the complete request. It includes every operator,
  loading boundary, and interaction in that workload.

Historical evidence is a fourth category: it can explain why a mechanism was
selected, but it is not a fresh result from your checkout. The
[performance appendix](./appendix-performance.md) labels each number as
component, product, control, historical, or unavailable.

![An evidence ladder from correctness to dispatch, component comparison, complete request, and a bounded decision.](./week2-performance-summary.svg)

## What the Accepted Evidence Says

The accepted Qwen3-4B evidence for the frozen successor found cumulative
complete-request latency improvements of **10.427%** at 128/128, **9.996%** at
512/128, **15.375%** at 2K/16, and **14.193%** at 2K/128 versus all mechanisms
off. Throughput relative to the matched full-MLX control was approximately
0.804, 0.822, 0.829, and 0.769. The 80% direction was met on the first three
rows and missed at 2K/128.

Those are supplied accepted measurements, not a command transcript for the
current learner checkout. The 2K/512 and 8K/128 product rows are unavailable
after environmental contamination; no component result is substituted for
them. Read the appendix before making a broader claim.

## What Is Selected—and What Is Not

The `selected` checkpoint keeps three measured mechanisms: request-bounded KV
capacity, register-cached RMSNorm, and tiled dense prefill attention. Packed W4,
SIMD matrix prefill, RoPE, and SwiGLU remain the cumulative path that makes those
checkpoints runnable.

Single-query decode attention remains the readable baseline. A prior bounded
single-kernel decode experiment is not a current checkpoint or Week 2
optimization. A prior reduction-splitting projection experiment is also
retired. Neither should appear in a current command, TODO, or selection claim.

Week 2 still optimizes one request. It does not add batching, paging, request
scheduling, or a production serving policy. Those are separate later-course
concerns.

## Precision and Fallbacks

Keep BF16 activations, scales, biases, cache entries, and model-facing outputs.
Packed W4 codes use `uint32`; reductions and online-softmax state accumulate in
FP32. Compare alternate reduction schedules with the supplied tolerances rather
than requiring bit-identical results.

Readable implementations are controls and fallbacks, not throwaway code. They
cover unsupported shapes, make failures easier to localize, and give every
optimized path a causal comparison.

[Optional static profiling guidance](./week2-advanced-profiling.md) explains how
to interpret supplied evidence without reviving retired command selectors.

{{#include copyright.md}}
