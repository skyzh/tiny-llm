# Task 346 bounded loader successor

## Exact boundary

- Steel baseline: commit
  `1078c08130fa0648fe45f11bf32ca33a7ecfdd58`, tree
  `164b5f9c867b51458a6dc3a85c197fd3e4322774`.
- Previously reviewed zero-Steel Draft: commit
  `84f6f4f9e9ae0e8b3bfee458569b063464ea0e3c`, tree
  `43300470ae50c5b99016b78d9725a00f32c9a91e`.
- Measured loader successor: commit
  `a97676aabb540826fb66799f631718a72d05c6c4`, tree
  `860a22c47a577e73519cc4f21d125b1fd8f4f88d`.
- Tuner task #352 report attachment:
  `437d6d2d-19a3-4766-beb0-57fff2ac77fe`, SHA-256
  `c874e90aa0d2b4b6d2b61e97c5df7a1d1e7f19242c7c63b0e1833699df54321f`.
- Tuner task #352 raw attachment:
  `43bdf915-4288-4ad4-835d-d2e7c7ea5cc7`, SHA-256
  `e86c7e26e12ea380d5ffe395dca06beee433798db325af508a38b27510e2de6f`.

The source delta from the reviewed Draft is exactly two files. The cooperative
loader now computes one row and starting column per thread, selects a
branch-free full-tile copy or the existing zero-filling edge-safe copy, and
offers one explicit 16-byte copy option. Only the Week 2 32x32 activation
loader enables that option; paged Q/K/V reuse the scalar full/safe paths.

The direct-MMA wrapper, paged-attention source, C++ dispatch, page translation,
manual cross-page fallback, masks, tests, starter, book, navigation,
publication files, and prior evidence are byte-identical to reviewed commit
`84f6f4f9`. The prior benchmark tree ledger remains SHA-256
`39f4f16b2743d372b122c1464b701cbaf6eb92aa076f450c1b29cd19c2baf367`.

## Correctness

The reference extension rebuilt from the measured commit. The focused Week 2
Day 6 suite passed 17 tests with 3 model-availability skips; Week 3 Day 5
passed all 4 contiguous/cross-page L=9/L=65 cases. The complete reference
suite passed **489 tests with 8 intentional model-availability skips**.

Every paged benchmark command independently matched grouped dense attention at
BF16 `rtol=atol=0.02`; the maximum absolute error was `0.015625`. All source
records were tracked-clean. Every serving command used request-trace SHA-256
`8365c7baa9f70ea15ecfe1bd55e9ce647dd66201af984b12d4eddce2f9d41720`.

## Frozen method

Both clean worktrees used the same `pdm.lock` SHA-256
`3e7c81ed0a7334188f5c61530529b03dcc927698d6448924cf0331c23005287f`,
the same Python 3.12.13 environment and model snapshot, and separately rebuilt
reference extensions. The paged harness remained SHA-256
`2a932b9aa9291cc0e379f5e7bc8aad5b097ed852da6f6a8820ee2cde702140d3`.

The original frozen comparison contract was reused without changing flags or
sample counts. Week 2 and paged operator commands ran in balanced whole-command
orders `Steel, successor, successor, Steel` then
`successor, Steel, Steel, successor`, with five untimed seconds between
commands. Serving used the first four-position order. Each Week 2 command
contains four forward/reverse shape sweeps, 12 warm-ups, and 60 synchronized
samples per case. Each paged command contains 600 conditioning calls per
shape, four forward/reverse sweeps, 12 warm-ups, and 60 synchronized samples
per case. Each serving command contains one warm-up and four fresh-process
samples.

The first Week 2 block showed a roughly 2.5% head-correlated MLX-control shift,
so it is retained as raw evidence but rejected for the decision table under the
contract's drift rule. A second complete balanced block used identical flags;
its MLX controls differed by only +0.08% at context 128 and -0.06% at context
512. The table below uses that confirmation block. Values are medians of four
independent command medians; lower latency is better.

## Results

| Week 2 surface | Steel (us) | Successor (us) | Delta |
|---|---:|---:|---:|
| SIMD, context 128 | 239.03 | 248.11 | +3.80% |
| Split-K, context 128 | 237.44 | 247.95 | +4.43% |
| MLX control, context 128 | 249.31 | 249.51 | +0.08% |
| SIMD, context 512 | 485.11 | 518.98 | +6.98% |
| Split-K, context 512 | 485.15 | 518.90 | +6.96% |
| MLX control, context 512 | 482.15 | 481.84 | -0.06% |

The loader successor therefore removes most of the reviewed Draft's Week 2
gap, but it does not close the separate direct-MMA gap identified by task #352.
The remaining slowdown is consistent for both SIMD and Split-K in the
drift-controlled confirmation block.

| Paged-prefill surface | Steel (us) | Successor (us) | Median delta |
|---|---:|---:|---:|
| L=9 | 199.98 | 202.59 | +1.31% |
| L=65 | 261.85 | 263.86 | +0.77% |

These paged medians are not a categorical regression. Command-level values
spanned 154.54–201.65 us for Steel and 170.25–205.92 us for the successor at
L=9, and 217.44–262.94 us versus 233.60–267.90 us at L=65. Direction reversed
across balanced positions, matching task #352's observed process-level clock
drift. The defensible conclusion is **mixed/noisy and within observed drift**.

Serving throughput is higher-is-better. Prefill is the primary integration
metric because the changed code runs only on prefill paths.

| Serving metric | Steel | Successor | Delta |
|---|---:|---:|---:|
| Prefill tokens/s | 3330.09 | 3085.53 | -7.34% |
| Output tokens/s | 160.75 | 156.93 | -2.37% |
| Decode tokens/s | 278.45 | 282.22 | +1.35% |
| Requests/s | 5.023 | 4.904 | -2.37% |

The decode observation is secondary and noisy because direct decode does not
use the changed loader. Serving prefill improves substantially over the prior
Draft's frozen -13.35% result, but this same-session successor remains 7.34%
below Steel.

## Recommendation

The bounded changes are correct, preserve the course-owned row-chunk model,
and recover most of the loader-specific regression. They do not close the
remaining Week 2 direct-MMA or end-to-end serving-prefill gaps. The conservative
recommendation remains **do not mainline this exact successor**. If the
course-owned zero-Steel direction is pursued further, the direct-MMA wrapper
should be a separately authorized, causal experiment rather than an expansion
of this loader patch.

The 28 successor evidence JSON files and their SHA-256 ledger accompany this
report. The earlier 20-file reviewed evidence corpus remains unchanged.
