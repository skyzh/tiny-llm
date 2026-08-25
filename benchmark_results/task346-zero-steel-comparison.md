# Task 346 zero-Steel comparison

## Exact boundary

- Steel base: commit `1078c08130fa0648fe45f11bf32ca33a7ecfdd58`, tree
  `164b5f9c867b51458a6dc3a85c197fd3e4322774`.
- Zero-Steel implementation: commit
  `69c4df4aa187088bdf6b78f3f1fc46be6107a0e7`, tree
  `17dd72ea5e818cb394d11647a34d030f0ba123d7`.
- Accepted comparison contract:
  `task346-zero-steel-baseline-contract.md`, SHA-256
  `a613600125d2b0a25bada656931134191b128600294fb0dbe089548ab232c5ea`.
- Frozen paged-prefill harness: `task346-paged-prefill-operator.py`, SHA-256
  `2a932b9aa9291cc0e379f5e7bc8aad5b097ed852da6f6a8820ee2cde702140d3`.

Both worktrees were tracked-clean, used the same `pdm.lock` blob and shared
Python 3.12.13 environment, and were built before timing on the same AC-powered
Apple M4 Pro host.  All JSON outputs were written outside the worktrees until
the full comparison was complete.

## Method

For each operator suite, whole-command order was:

1. Steel, zero, zero, Steel.
2. Zero, Steel, Steel, zero.

There were five untimed seconds between command invocations.  Each Week 2
command contained four forward/reverse shape sweeps, 12 warm-ups, and 60 timed
rounds per case.  Each paged-prefill command contained 600 conditioning calls
per shape, four forward/reverse shape sweeps, 12 warm-ups, and 60 timed calls
per case.  The serving suite used one Steel, zero, zero, Steel block; every
command ran four fresh processes after one warm-up.  The tables use the median
of the independent command-level medians for each head.  The 20 corresponding
JSON files preserve every inner sample, execution order, host/configuration,
source identity, and correctness result.

## Results

Operator latency is in microseconds and lower is better.  A positive delta is
a zero-Steel regression.

| Surface | Steel | Zero-Steel | Delta |
|---|---:|---:|---:|
| Week 2 SIMD, context 128 | 238.81 | 269.21 | +12.73% |
| Week 2 split-K, context 128 | 239.17 | 268.85 | +12.41% |
| Week 2 MLX control, context 128 | 249.83 | 250.83 | +0.40% |
| Week 2 SIMD, context 512 | 485.53 | 643.40 | +32.51% |
| Week 2 split-K, context 512 | 484.73 | 637.03 | +31.42% |
| Week 2 MLX control, context 512 | 482.32 | 489.54 | +1.50% |
| Week 3 paged prefill, L9 | 200.34 | 207.28 | +3.46% |
| Week 3 paged prefill, L65 | 263.80 | 272.58 | +3.33% |

Serving throughput is higher-is-better.  Prefill is the end-to-end primary
metric because the replaced primitives are prefill paths.

| Serving metric | Steel | Zero-Steel | Delta |
|---|---:|---:|---:|
| Prefill tokens/s | 3335.69 | 2890.27 | -13.35% |
| Output tokens/s | 162.78 | 162.16 | -0.38% |
| Decode tokens/s | 281.45 | 309.12 | +9.83% |
| Requests/s | 5.087 | 5.067 | -0.38% |

All eight paged-prefill command files matched grouped dense attention at BF16
`rtol=atol=0.02`; the maximum absolute error was `0.015625`.  Every serving
command used the same request trace SHA-256
`8365c7baa9f70ea15ecfe1bd55e9ce647dd66201af984b12d4eddce2f9d41720`.
The decode gain is a secondary noisy observation: direct decode did not use the
removed Steel primitives and therefore cannot offset the prefill regression.

## Recommendation

Keep this Draft as a readable correctness prototype, but do not put this exact
implementation on the mainline course path.  Direction is consistent across
both changed operator families and the end-to-end primary metric: the
course-owned implementation is slower, materially so for Week 2 at context
512 and for serving prefill.  A future mainline candidate should retain the
small explicit loader interface while first closing the direct-MMA and
quantized-prefill performance gap under the same interleaved contract.
