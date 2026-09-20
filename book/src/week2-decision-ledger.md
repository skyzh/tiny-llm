# 🚧 Week 2 Decision Ledger

Keep this record beside your benchmark output. Begin with the working Week 1
model from [Measurement](./week2-02-benchmark-profile.md), then add a row after
each completed checkpoint. The record is your work; the benchmark, attribution
runner, test files, and checkpoint controls are supplied.

## Record One Comparison at a Time

For every row, save the model and source version, device/software, checkpoint,
phase, input/output lengths, prefill-logit mode, warmups, repetitions, and output
files. Explain why the baseline and candidate do the same requested work.

| Change | Completed checkpoint | Matched control | Your result and decision |
|---|---|---|---|
| Cache reuse and capacity-backed append | `kv-cache` | `week1`, plus the cache storage witness | Pending your measurement |
| Packed W4 decode | `quantized-matvec` | `kv-cache`; vanilla W4 for local arithmetic | Pending your measurement |
| Matrix prefill | `simd-matmul` | `quantized-matvec` | Pending your measurement |
| RMSNorm | `rmsnorm` | `simd-matmul` | Pending your measurement |
| RoPE | `rope` | `rmsnorm` | Pending your measurement |
| Elementwise SwiGLU | `swiglu` | `rope` | Pending your measurement |
| Shared-input QKV | `shared-input-qkv` | Same checkpoint with `--disable-week2-shared-input-qkv` | Pending your measurement |
| Shared gate/up and SwiGLU | `shared-input-gate-up-swiglu` | Same checkpoint with `--disable-week2-shared-input-gate-up-swiglu` | Pending your measurement |
| I/O-aware dense attention | `io-aware-dense-attention` | Same checkpoint with `--disable-week2-io-aware-dense-attention` | Pending your measurement |

Correctness, supported dispatch, component timing, and complete-request timing
answer different questions. Record each where it applies. A lower operator time
with no request improvement is still useful information; explain why it does
not yet justify a product decision.

- **Keep:** correctness and dispatch checks pass, and repeated matched results
  support using the change for the stated workload.
- **Reject:** the change is incorrect or the matched result does not justify it
  for that workload. Preserve the control and identify the failed expectation.
- **Inconclusive:** variability, missing observations, or insufficient samples
  prevent a choice. Name the next measurement that would resolve it.

These are fields for your results, not decisions already made for the current
implementation. A supported shape is not a measured performance range.

## Close the Cumulative Route

The final incoming checkpoint is `io-aware-dense-attention`. There is no new
kernel in this closing step. Reuse the supplied final gate first:

```bash
pdm run test --week 2 --day 7
```

After all earlier chapter gates pass, you can run the complete product ladder:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week1 --variant week2-kv-cache \
  --variant week2-quantized-matvec --variant week2-simd-matmul \
  --variant week2-rmsnorm --variant week2-rope --variant week2-swiglu \
  --variant week2-shared-input-qkv \
  --variant week2-shared-input-gate-up-swiglu \
  --variant week2-io-aware-dense-attention --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-final.json
```

This ladder summarizes cumulative checkpoints. Use the chapters' same-checkpoint
disable pairs when isolating the three final mechanisms; the full-MLX model is
a separate implementation comparison, not an operator-disable control.

Explain the final outcome in terms of the request you measured: where time went,
which change affected it, which result did not carry through to the product,
and what still needs investigation. Then continue to
[Week 3](./week3-overview.md), where scheduling and paging introduce different
workloads and require new comparisons.

The [historical performance appendix](./appendix-performance.md) preserves
measurements of earlier source trees. It does not fill the pending rows above.

{{#include copyright.md}}
