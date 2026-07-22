# 🚧 Week 2 Day 7: End-to-End Optimization

> 🚧 This newly introduced chapter is a work in progress.

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
- `qwen3_week3.py` imports Week 2 model components, reuses the cache interface,
  and adds paged cache storage and serving mechanisms.

Later weeks must not overwrite earlier files. The starter and reference trees
expose the same module and function names so each checkpoint remains runnable.

## Validate Correctness

```bash
pdm run test --week 2 --day 7
```

Check full-sequence logits with `logits_to_keep=None`, last-position logits with
`logits_to_keep=1`, and incremental decoding against the MLX model.

## Measure the Result

Repeat the matched benchmark from Day 1. The required path should reach at least
80% of MLX decode throughput on the same machine. This is a target for the
course-owned path, not permission to replace a slow operator with the matching
MLX implementation. If the result misses the target, profile it, improve the
kernel, and report the measured gap honestly.

Before accepting the result, inspect the Week 2 source for accidental shortcuts.
The model may use `mlx_lm` for loading and `mlx.core` for arrays, device
execution, synchronization, and extension dispatch, but its learned operators
must resolve to the Python/C++/Metal implementations from this course.

## Expected Performance Contribution

**Estimated decode improvement: 5-15% beyond the operator kernels, with an
overall goal of at least 80% of matched MLX throughput.** Last-token logits and
graph cleanup avoid work outside the transformer blocks. Report the final
measured ratio; do not present this estimate as a substitute for it.

The current M1 Pro checkpoint uses Qwen3-0.6B-MLX-4bit, a 128-token prompt, 64
timed decode tokens, and two warmups. Across three otherwise idle runs, median
decode throughput was 173.7 tok/s for the course-owned path and 319.9 tok/s for
MLX, or 54.3%. This work-in-progress result does not satisfy the 80% target; it
identifies the remaining kernel-optimization work without replacing those
kernels with MLX-provided implementations.

{{#include copyright.md}}
