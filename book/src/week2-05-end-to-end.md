# Week 2 Days 6-7: End-to-End Decode Optimization

The final step is to connect the optimized operations and remove graph work
that does not contribute to the next token.

## Keep Only Needed Logits

Prefill computes hidden states for the whole prompt, but generation samples only
from the final position. Add an optional `logits_to_keep` argument to the Week 2
and Week 3 models. Slice hidden states before the final norm and vocabulary
projection:

```python
if logits_to_keep is not None:
    h = h[:, -logits_to_keep:, :]
```

Generation and batching should request one position. Callers that need all
positions, including correctness tests and prompt scoring, pass `None`.

## Preserve the Incremental Layers

- `qwen3_week1.py` keeps readable Python operators and dequantized weights.
- `qwen3_week2.py` uses quantized weights and `week2_kernels.py`.
- `qwen3_week3.py` imports Week 2 model components, then adds serving caches and
  paged attention.

Later weeks must not overwrite earlier files. The starter and reference trees
expose the same module and function names so each checkpoint remains runnable.

## Validate Correctness

```bash
pdm run test --week 2 --day 6
```

Check full-sequence logits with `logits_to_keep=None`, last-position logits with
`logits_to_keep=1`, and incremental decoding against the MLX model.

## Measure the Result

Repeat the matched benchmark from Day 1. The required path should reach at least
80% of MLX decode throughput on the same machine. If a custom kernel misses the
target, keep it behind an explicitly named educational function and use the
production primitive in `quantized_matmul`.

The reference solution follows that rule: `quantized_matmul` is the native fast
path, while `quantized_matvec_custom` exposes the SIMD Metal kernel for direct
M=1 and M=8 experiments.

{{#include copyright.md}}
