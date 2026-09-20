# 🚧 Week 2: Make One Request Faster

Your Week 1 model can generate text. Give it a longer prompt, however, and the
wait changes. Generating the next token also repeats work you have already done.
This week you will measure those costs, remove one at a time, and check whether
the complete request benefits.

Begin with [Measurement](./week2-02-benchmark-profile.md). You can run the Week 1
product before writing any Week 2 kernels. Keep that first result: it gives the
cache exercise a concrete before-and-after comparison.

The route follows changes to the model, rather than a fixed number of days:

| Step | What you change | Completed checkpoint | Supplied test selector |
|---|---|---|---|
| [Measurement](./week2-02-benchmark-profile.md) | Choose a workload and record its timing contract | Working Week 1 model | `--week 2 --day 2` |
| [KV cache](./week2-01-kv-cache.md) | Reuse past K/V and append within reserved capacity | `kv-cache` | `--week 2 --day 1` |
| [Packed W4 decode](./week2-03-quantize-model.md) | Keep projection weights packed through the dot product | `quantized-matvec` | `--week 2 --day 3` |
| [Matrix prefill](./week2-05-simd-matrix-prefill.md) | Reuse tiles across prompt rows | `simd-matmul` | `--week 2 --day 5` |
| [Compact primitive lab](./week2-04-fused-model-kernels.md) | Fuse RMSNorm, RoPE, then elementwise SwiGLU | `rmsnorm` → `rope` → `swiglu` | `--week 2 --day 4` |
| [Shared-input fusion](./week2-06-operator-lab.md) | Share activation loads across QKV, then gate/up | `shared-input-qkv` → `shared-input-gate-up-swiglu` | `--day 6`, then a focused `--day 7` slice |
| [I/O-aware dense attention](./week2-07-split-k-prefill.md) | Stream softmax state instead of storing the score matrix | `io-aware-dense-attention` | `--week 2 --day 7` |

The `--day` numbers are stable test-file selectors. Follow the chapter order in
the navigation; matrix prefill now comes before the primitive lab. Kernel work
can take substantially longer than a Week 1 exercise.

## Keep a Complete Request in View

A faster isolated operator is a useful lead. To decide whether to use it, you
also need to know whether the model selects it and whether the same request gets
faster. Each chapter therefore ends at a runnable checkpoint with an earlier
checkpoint or a disable flag as its control.

![The evidence ladder: correct output, eligible model dispatch, matched operator comparison, matched complete request, then a keep, reject, or inconclusive decision.](./week2-performance-summary.svg)

The benchmark and attribution tools are supplied. You own the cache, Python
wrappers, native primitives, Metal arithmetic, and their integration into
`Qwen3ModelWeek2`. Empty starter bodies are intentional. An early test should
identify the missing work; a completed checkpoint must exercise your solution.
Always pass `--solution tiny_llm`: several measurement tools otherwise select
the completed reference.

After changing extension source, rebuild with `pdm run build-ext`. The chapter's
`pdm run test` command copies the supplied test into `tests/` and runs it against
your implementation. A focused test is useful during construction; run the full
named gate when the chapter says the checkpoint is complete. An optional model
test skipped for missing weights does not establish model correctness.

## Let the Workload Choose the Next Experiment

During one-token decode, projection dimensions are fixed by the model, while
the attention operation reads a prefix that grows with context. During prefill,
many activation rows can reuse the same weight tile. Those differences give you
separate reasons to investigate cache traffic, projection schedules, and
attention storage. In the diagram, QKVO means the query, key, value, and output
projections; MLP names the feed-forward block, including its SwiGLU activation. They do not predict which implementation wins on your GPU.

![Two sources of work for a decode token: QKVO and MLP projections keep the model's fixed dimensions, while attention reads more K/V as context grows. Measure the resulting balance; no crossover length is assumed.](./week2-kernel-profile.svg)

Keep BF16 activations, scales, biases, cache entries, and model-facing outputs;
packed W4 codes use `uint32`. Accumulate reductions in FP32, then return the
required output dtype. Compare results with the supplied tolerances rather than
requiring different reduction schedules to be bit-identical.

The chapters provide experiments, not predetermined KEEP decisions. Record the
workload, the result, and what would make you reverse your choice. The
[decision ledger](./week2-decision-ledger.md) keeps those records together.
[Metal capture](./week2-advanced-profiling.md) is an optional tool for questions
that remain after the portable measurements.

By the final checkpoint, the single-request model has packed weights,
shape-aware projections, shared-input experiments, and dense attention that
avoids storing its scores. [Week 3](./week3-overview.md) changes request scheduling
and cache layout. A dense cache and a dense-attention kernel alone do not supply
paging or continuous batching.

{{#include copyright.md}}
