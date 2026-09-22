# 🚧 Week 2 Day 7: Select the Cumulative Path

Day 7 does not add another kernel. It turns the week's mechanisms into an
auditable decision: which optimizations belong in the completed model, which
controls remain readable, and which claims the evidence does not support.

The public checkpoint is `selected`. It contains exactly these selected
mechanisms:

1. request-bounded KV capacity;
2. register-cached RMSNorm;
3. tiled dense prefill attention.

Packed W4, SIMD matrix prefill, RoPE, and SwiGLU remain in the cumulative path.
They are required checkpoint work, but the three controls above are the
independently selectable mechanisms evaluated in the accepted progression.

## First Diagnostic: Inspect the Feature Set

```bash
pdm run test --week 2 --day 7 -- -k exactly
```

The witness checks the immutable checkpoint map rather than timing a model. A
failure means selection drift: do not patch around it with a special command or
hidden default.

`Qwen3ModelWeek2` also accepts three constructor controls for focused tests and
experiments:

```text
use_bounded_kv_capacity
use_register_cached_rms_norm
use_tiled_prefill_attention
```

Each accepts `True`, `False`, or `None`; `None` inherits the checkpoint feature.
`mechanism_controls` exposes the resolved state. Low-level mechanisms remain
independently default-off before their checkpoint, while the completed
no-argument model defaults to `selected` for compatibility.

## Prove Controls and Defaults

```bash
pdm run test --week 2 --day 7 -- -k independently
pdm run test --week 2 --day 7 -- -k default_model
pdm run test --week 2 --day 7
```

The full gate also confirms that older experimental labels are rejected as
unknown checkpoint names. They are not aliases, current commands, or TODOs.
Single-query decode continues through the readable attention baseline.

## Run the Selected Product

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint selected --model qwen3-0.6b --max-tokens 16
```

If a selected mechanism is ineligible, use its documented readable fallback:
concatenating dense cache before Day 2, fixed-width RMSNorm above dimension
4096, or readable grouped attention below the tiled prefill boundary. A fallback
is part of the contract, not a silent failure.

## Read the Accepted Product Evidence

The frozen accepted campaign used Qwen3-4B and matched all-off/full-MLX controls.
Medians below are complete-request results; `MLX ratio` is throughput divided by
the identical-feed full-MLX denominator.

| Prompt/output | Selected total-latency gain vs all-off | MLX ratio | 80% direction |
|---:|---:|---:|---|
| 128/128 | 10.849% | ≈0.806 | met |
| 512/128 | 10.347% | ≈0.822 | met |
| 2K/16 | 15.402% | ≈0.828 | met |
| 2K/128 | 14.126% | ≈0.770 | missed |
| 2K/512 | unavailable | unavailable | no verdict |
| 8K/128 | unavailable | unavailable | no verdict |

The two unavailable rows were stopped after environmental contamination. They
are not zeros, failures, or invitations to fill the cells from component
benchmarks. The accepted campaign also found RMSNorm improvements of 71.36% at
2K and 83.81% at 8K, and tiled-prefill improvements of 54.07% at 2K and 54.89%
at 8K. Those component results explain mechanisms; they do not supply the
missing product rows.

## Make Your Own Bounded Decision

Compare the immediate product predecessor and the selected checkpoint on one
identical local request:

```bash
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint tiled-prefill --model qwen3-0.6b --max-tokens 16
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint selected --model qwen3-0.6b --max-tokens 16
```

These two public checkpoints currently have the same cumulative feature set;
the comparison verifies the named handoff rather than isolating a new kernel.
Use the independent constructor controls only in code/tests when you need a
single-mechanism ablation.

Close your Week 2 ledger with one row per mechanism:

| Field | What to record |
|---|---|
| Workload | model, prompt/output bound, device, software, and control |
| Correctness | focused and completed gate result |
| Routing | cache or dispatch counters proving the intended path |
| Component | same-shape operator comparison, or `not measured` |
| Product | matched complete-request observation, or `unavailable` |
| Decision | keep, reject, or inconclusive for this workload |
| Falsifier | the result or shape that would reverse the decision |

Do not add percentages from different denominators. Do not infer hardware
occupancy from source geometry or elapsed time. Do not convert an unavailable
row into an estimate.

Week 2 ends with a faster measured path for one request and explicit readable
controls. It makes no claim about batching, paging, scheduling, or production
serving policy.

The [performance appendix](./appendix-performance.md) preserves the exact
accepted evidence categories and starts a fresh decision ledger for this
successor.

{{#include copyright.md}}
