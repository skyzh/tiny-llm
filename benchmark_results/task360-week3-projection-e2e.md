# Task 360 Week 3 projection seam

## Exact boundary

- Reviewed predecessor: commit
  `22dffc1cf822cbdca618de926edf204270a30027`, tree
  `3f4d9b7d8aa40813e77d25eef2f6cd3ec678ccd2`.
- Measured successor source: commit
  `170211be3503c0ec0b1fa75bbb3b0c23a86bd3ac`, tree
  `25e588de248692b711e04014ba43e85326d21393`.
- Tuner task #359 report attachment:
  `0f4cfaae-b754-4ef5-9a97-74491b0349cf`, SHA-256
  `3aba6f8eedfb9b0a5800e76c2f26797eaab6a110b40a349fd1bfd9ead0decf4f`.
- Tuner task #359 raw attachment:
  `fd2cf5ec-83d0-4693-b064-b3edb8f02da3`, SHA-256
  `8d70b92afe5516fb9cf189250943d0d632ad8d43e523d242d3d8730ce647076d`.

The successor keeps the Week 2 zero-Steel implementation as the default and
adds one explicit projection selector carried by the quantized weight wrapper.
Canonical dense Week 3 model construction enables MLX quantized projections;
the Week 3 scheduler factory does the same for its Week 2 dense-cache model.
The optional MoE path remains on its separate inherited projection contract.
The benchmark-only `--week3-inherit-course-projections` flag provides a causal
ablation of the old inheritance behavior.

The cooperative loader, direct-MMA wrapper, paged-attention kernels and
dispatch, block-table translation, cross-page fallback, masks, KV-cache and
scheduler algorithms are byte-identical to the reviewed predecessor. Week 2
continues to default to the course-owned projections. No publication or
navigation file changed.

## Correctness and causal checks

- Focused Week 3 Day 1 and Day 3: 26 passed, 1 model-availability skip.
- Complete Week 3 reference suite: 101 passed, 1 skip.
- Complete Week 2 reference suite: 53 passed, 3 skips.
- Starter/reference interface synchronization: 12 passed after rebuilding
  both extensions.
- Complete reference suite: **496 passed, 8 intentional skips**.
- Benchmark harness tests: 68 passed.
- Scoped Ruff check and format check: clean.
- Manual mdBook build: passed with only the existing preprocessor-version
  warnings.

Three exact mutations demonstrated that the tests observe the seam: disabling
the Week 3 scheduler-factory flag failed its focused assertion; changing the
Week 3 constructor default to inherited projections failed the dense-model
flag assertion; and bypassing the quantized projection selector failed the
mocked-MLX causal test by entering the course extension. The correct code was
restored and all four focused selector/boundary tests passed.

The Day 3 coverage exercises tied and untied dense weights, prompt and decode
calls, and a page-boundary case while recording the MLX projection calls. A
paired ablation checks identical cache/attention outputs between the explicit
seam and inherited projection modes.

## Frozen method

All runs used the same tracked-clean source, local Qwen3 0.6B snapshot, Python
3.12.13 environment, MLX 0.32.0, GPU, and `prefill_logits=last` on the same
Apple-silicon host. Commands ran sequentially in fresh processes.

The Week 2 ladder uses two alternating samples each for the SIMD, Split-K, and
MLX checkpoints. Chunked Week 3 Day 1/2 uses balanced whole-command order
`seam, inherited, inherited, seam` at prefill steps 512 and 128. Day 3 uses
`seam, inherited, MLX, MLX, inherited, seam`. Serving uses
`seam, inherited, inherited, seam`. Each native E2E command includes one
warm-up. Within each workload, every command consumed an identical request
trace; the manifest records its SHA-256.

Values below are medians of two fresh-process samples; higher is better.

## Results

| Week 2 checkpoint | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| 2.6 SIMD matrix prefill | 4221.93 | 228.12 |
| 2.7 Split-K prefill | 4200.07 | 237.52 |
| MLX | 5477.70 | 277.92 |

This is a preservation check rather than an old/new comparison: canonical
Week 2 does not enable the new selector, and its source path remains the
course-owned zero-Steel implementation.

| Chunked Day 1/2 | Seam prefill tok/s | Inherited prefill tok/s | Delta | Seam output tok/s | Inherited output tok/s | Delta |
|---|---:|---:|---:|---:|---:|---:|
| prefill step 512 | 4761.96 | 4304.19 | +10.64% | 167.48 | 149.65 | +11.91% |
| prefill step 128 | 4155.09 | 3718.63 | +11.74% | 156.67 | 140.18 | +11.76% |

Both balanced positions favored the explicit Week 3 seam at both chunk sizes.

| Day 3 dense path | Prefill tok/s | Output tok/s | Decode tok/s |
|---|---:|---:|---:|
| Explicit projection seam | 4650.25 | 187.50 | 254.56 |
| Inherited Week 2 projections | 4145.89 | 160.50 | 214.18 |
| Full MLX control | 5479.54 | 229.47 | 317.27 |

The seam is +12.17% prefill, +16.82% output, and +18.86% decode versus the
inherited zero-Steel projection path. It remains 15.13% below full MLX on
prefill, which is expected because attention/cache/paging stay course-owned.

| Serving metric | Seam | Inherited | Delta |
|---|---:|---:|---:|
| Prefill tokens/s | 2842.50 | 2638.73 | +7.72% |
| Output tokens/s | 162.78 | 148.77 | +9.42% |
| Decode tokens/s | 299.03 | 264.58 | +13.02% |
| Total tokens/s | 1361.81 | 1244.62 | +9.42% |

Both balanced serving positions favored the explicit seam on every reported
throughput metric. The absolute gain is smaller than in the single-request Day
3 run, but the direction is consistent.

## Recommendation

The explicit dense Week 3 projection seam is correct, narrow, and causally
measurable. It preserves Week 2's zero-Steel teaching path while preventing
Week 3 cache, attention, paging, and serving benchmarks from inheriting the
known Week 2 projection cost. The evidence supports keeping this successor on
PR #277 for fresh independent review and Chi's mainline decision. It does not
change the earlier conclusion that the all-zero-Steel implementation itself
should not be mainlined as the canonical Week 3 performance path.

The 20 JSON evidence files and their SHA-256 ledger accompany this report.
