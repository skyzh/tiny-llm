# Tiny task #347: exact-main Steel baseline and zero-Steel comparison contract

## Decision boundary

This is a read-only baseline and methodology artifact for task #346. It does
not evaluate an unpublished zero-Steel implementation and therefore makes no
speedup, no-regression, equivalence, or merge recommendation. Absolute timings
below are an anchor for this host; the eventual decision must come from an
interleaved same-session comparison of the exact Steel and zero-Steel heads.

Repository base: commit
`1078c08130fa0648fe45f11bf32ca33a7ecfdd58`, tree
`164b5f9c867b51458a6dc3a85c197fd3e4322774`. The tracked checkout was clean for
all measurements.

## Required-solution Steel inventory

Only two reference-solution Metal files contain `mlx::steel`:

1. `src/extensions_ref/src/quantized_matmul.metal`
   - Includes `mlx/backend/metal/steel/gemm/gemm.h` and
     `mlx/backend/metal/steel/gemm/mma.h`.
   - The shared W4A16/G128 SIMD-group helper instantiates
     `BlockMMA<T, OutT, 32, 32, 32, 2, 2, false, true, 40, 40>` and
     `BlockLoader<T, 32, 32, 40, 1, 128>`.
   - `BlockLoader` stages the activation tile; weight unpack/dequantization is
     course code. `BlockMMA::mma` performs the tile multiply and
     `store_result_safe` publishes full or partial output tiles.
   - Both the ordinary SIMD-group kernels and the split-K kernels call this
     same helper. The direct decode matvec does not use Steel.

2. `src/extensions_ref/src/paged_attention.metal`
   - Includes `mlx/backend/metal/steel/attention/loader.h`.
   - Only the BF16, head-dimension-128, `L > 8` paged-prefill kernel uses Steel:
     `BlockLoaderT` stages Q (64x128) and single-page K/V tiles (32x128).
   - Page-table translation, non-single-page K/V loading, QK/PV
     `simdgroup_matrix` multiplication, masking, online softmax/rescaling, and
     output store are course-owned code. The `L <= 8` direct-decode path and
     scalar fallback do not use Steel.

There are no other required-reference `mlx::steel` occurrences. This is also
the preservation boundary: a zero-Steel patch may replace those staging/MMA
uses, but should not claim to have rewritten the already course-owned paging,
softmax, or direct-decode mechanics.

Source SHA-256 values:

- `pdm.lock`: `3e7c81ed0a7334188f5c61530529b03dcc927698d6448924cf0331c23005287f`
- `quantized_matmul.metal`: `812a1712df29be839a782d28fb807c27779958f414dd3d68c583c61497942858`
- `paged_attention.metal`: `eb12483f2921833ddb2c201df6b86aba9f4dbbdc2d4736e8dba769a36ddfdfc2`
- `bindings.cpp`: `9b5e8e2d910adf8345a041f0e60c30226a40f2d16879e0c2b72527e2675eb668`

## Current Steel baseline

All timings use synchronized `mx.eval`, lower-is-better microsecond latency for
operators, and fresh-process medians for the end-to-end serving run.

| Surface | Exact configuration | Current Steel anchor |
|---|---|---|
| Week 2 W4A16/G128 Q projection | Qwen3-0.6B, BF16, contexts 128/512, ordinary and split-K, 4 forward/reverse shape sweeps, 12 warm-ups and 60 timed rounds per case | context 128: SIMD 205.7 us, split-K 205.6 us, MLX control 215.0 us; context 512: SIMD 442.6 us, split-K 442.0 us, MLX control 441.5 us |
| Week 3 paged BF16 prefill | B1, Hq16, Hkv8, D128, page size 32, prefix 64, causal query lengths 9/65, noncontiguous physical pages, 600 untimed conditioning calls per shape, 4 forward/reverse sweeps, 12 warm-ups and 60 timed calls per case | L9: 331.875 us median over 240 raw samples; L65: 553.500 us median over 240 raw samples |
| Direct-paged continuous batching | Qwen3-0.6B, 8 requests, batch 4, prompt 128..512, output 32, prefill step 128, seed 0, 1 warm-up, 4 fresh-process samples, 1 s internal cooldown | prefill 3602.884 tok/s; output 190.704 tok/s; decode 341.358 tok/s; 5.959 req/s |

The paged operator checked every constructed case against grouped dense
attention at BF16 tolerances `rtol=atol=0.02`; maximum observed absolute error
was at most 0.015625. The deterministic serving request-trace SHA-256 is
`8365c7baa9f70ea15ecfe1bd55e9ce647dd66201af984b12d4eddce2f9d41720`.
As a build/smoke check, the two focused SIMD matmul tests, all four split-K
tests, and both paged-FlashAttention tests passed: 8/8.

These baseline values show noticeable warm-up/DVFS history in the operator raw
runs, which is exactly why a later-head-only run is not a valid comparison.
Retain all raw samples; do not quote only the best repeat or compare a future
run directly with this table.

## Frozen commands

Use the same Python environment for both worktrees:

`/Users/skyzh/.slock/agents/fea96ef3-7fe1-4df2-83a9-cf900a4849c4/worktrees/tiny-llm-task205-day1-review/.venv/bin/python`

Build each clean head from its own `src/extensions_ref/` directory with:

```sh
CMAKE_ARGS="-DPython_EXECUTABLE=<python-above> -DPython_ROOT_DIR=<shared-venv-root> -DPython_FIND_VIRTUALENV=ONLY" <python-above> build.py
```

Then run these commands from each repository root, changing only the absolute
JSON output filename:

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 <python-above> -m benches.bench_week2_operators \
  --model qwen3-0.6b --solution tiny_llm_ref \
  --section prefill-projections \
  --context 128 --context 512 --context-repeats 4 \
  --warmup 12 --iterations 60 --prefill-projection q --include-split-k \
  --json-output <output>.json
```

```sh
PYTHONPATH=src:. HF_HUB_OFFLINE=1 <python-above> \
  /Users/skyzh/.slock/agents/fea96ef3-7fe1-4df2-83a9-cf900a4849c4/reports/task347-paged-prefill-operator.py \
  --conditioning 600 --warmup 12 --iterations 60 \
  --shape-repeats 4 --seed 0 --json-output <output>.json
```

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 <python-above> -m benches.bench_serving_progression \
  --model qwen3-0.6b --solution ref --variant paged \
  --num-seqs 8 --batch-size 4 \
  --min-input-len 128 --max-input-len 512 \
  --min-output-len 32 --max-output-len 32 \
  --prefill-step 128 --warmup 1 --repeats 4 --seed 0 \
  --offline --cooldown-seconds 1 --json-output <output>.json
```

The custom paged-prefill harness is itself frozen at SHA-256
`2a932b9aa9291cc0e379f5e7bc8aad5b097ed852da6f6a8820ee2cde702140d3`.
Do not use `benches.bench_week3_attention` for this decision: it fixes query
length at one and therefore dispatches the non-Steel direct-decode kernel.

## Same-session comparison order and decision rules

1. Pin and record the exact Steel and zero-Steel commit and tree SHAs. Require
   clean tracked status, the same `pdm.lock`, model snapshot, Python environment,
   compiler/toolchain, and benchmark-harness hash. Build both before timing.
2. Run on this same otherwise-idle AC-powered host. For each suite separately,
   use head order `Steel, zero, zero, Steel`, with five untimed seconds between
   whole command invocations. Never run all Steel samples first and all
   zero-Steel samples later.
3. Keep every raw JSON. For operator suites, repeat the four-position order a
   second time as `zero, Steel, Steel, zero`; this gives four command-level
   observations per head while balancing early/late position. For the slower
   end-to-end suite, the first four-position block already yields eight fresh
   process samples per head because each command contains four repeats.
4. Reject a run before comparing if configuration, execution order, model
   snapshot, trace hash, case correctness, or sample counts differ; if the MLX
   control shifts materially between heads, report environment drift and rerun
   rather than attributing it to the patch.
5. Operator primary metrics are the per-context median latency for `simd` and
   `split-k`, and the paged L9/L65 median latency. End-to-end primary metric is
   prefill tok/s; output tok/s, requests/s, and decode tok/s are secondary
   integration observations because the changed Steel paths are prefill paths.
6. Report signed per-head deltas and all command-level medians. Do not pool the
   hundreds of within-process operator timings as independent replicates, use a
   best repeat, invent a universal acceptable-regression threshold, or claim
   statistical significance from one host. A categorical performance claim is
   allowed only if direction is consistent across balanced command-level runs
   and the operator and end-to-end observations agree; otherwise report the
   result as mixed/noisy.
7. Performance never substitutes for correctness. The zero-Steel head must
   separately preserve ordinary/split-K partial tiles, identical model results,
   paged L9/L65 equivalence including noncontiguous pages, the manual
   non-single-page loader boundary, and unchanged dispatch/fallback behavior.

## Raw-output schemas and evidence files

- Week 2 operator JSON uses schema version 2: `metadata`, `configuration`, raw
  `runs` with per-implementation `samples_us` and rotated
  `measurement_orders`, plus the aggregate `summary`.
- Paged-prefill JSON uses schema version 1: `source`, `host`, harness hash,
  complete `configuration`, per-repeat correctness, raw `runs/samples_us`, and
  `summary`.
- Serving JSON contains `source`, `host`, `configuration`, variant definitions,
  fresh-process execution order, exact request trace and hash, every metrics
  sample, and medians.

Evidence SHA-256 values:

- `task347-steel-week2-operator.json`: `c83de10476a9f796d228b051a4d283744f017ee60a52a637cbe559031286e20f`
- `task347-steel-paged-prefill-operator.json`: `02f2480d7fac228f69559902e13f1973289a3ebabe2f7d5a160a40560b633bcd`
- `task347-steel-e2e-serving.json`: `79713751b6383c3258ba4583254f25b487bfe39b4f719caca99ef39a04781d18`

## Host/software metadata

- Mac mini `Mac16,11`, Apple M4 Pro, 14 CPU cores (10 performance + 4
  efficiency), 20 GPU cores, 64 GB physical/device memory; MLX architecture
  `applegpu_g16s`, recommended working set 55,662,788,608 bytes.
- macOS 26.5.2 build 25F84; Darwin 25.5.0 arm64; AC power.
- Python 3.12.13; MLX 0.32.0; mlx-lm 0.31.3; NumPy 2.2.6; nanobind 2.13.0;
  pytest 8.4.1.
- Xcode 26.6 build 17F113; Apple Metal 32023.883; Apple clang 21.0.0;
  CMake 4.2.3; Apple Git 2.50.1.
- Offline model `Qwen/Qwen3-0.6B-MLX-4bit`, cached snapshot
  `173234aa840d113125e9f2271100ddbaf16c9620` (BF16 scales, 4-bit weights,
  group size 128 as captured in the raw operator metadata).
- Current compiled baseline artifact hashes (diagnostic only, not expected to
  reproduce byte-for-byte across build paths): extension
  `04c865091be2e1d9caca96417abfdcc35510da25d63464378cdca274e29129bf`,
  dylib `7efee5c87cfde872d314973d39432f8bcc530fc0ae2a6e606e26c59fc5cdf948`,
  metallib `448c4c6ee71543e9a618e18545b8a504778d24c84e8ec9f64817672e57d71f46`.

No repository, GitHub, book, release, tag, or deployment state was changed.
