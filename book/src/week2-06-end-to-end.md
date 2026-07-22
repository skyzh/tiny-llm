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

## Optimization Ledger

Record both retained and rejected experiments. End-to-end effects overlap, so
the table uses representative ranges rather than pretending every row is
independently additive.

| Change | Decision | Representative effect and reason |
|---|---|---|
| Keep weights quantized | Keep | Several-fold decode gain over Week 1 by reducing weight traffic. |
| Two-column SIMD matvec | Keep | About +5.7% in its first isolated end-to-end ablation; shares activation loads across columns. |
| Eight-column vocabulary matvec | Keep | Best measured layout for the very wide tied output head; smaller projections use two columns to avoid register pressure. |
| Remove matvec activation barrier | Keep | Roughly 238.6 → 247-252 tok/s; a cache-hot 2 KB vector was cheaper to reread than to copy and synchronize in every threadgroup. |
| Vanilla → `simdgroup_matrix` prefill | Keep | About 1,005 → 2,052 prefill tok/s; 8×8 matrix fragments replace scalar dot products. |
| Direct quantized embedding | Keep | Roughly 20% lower isolated latency in a representative run, about 1% end to end; gathers and dequantizes without intermediate arrays. |
| 256-thread RMSNorm | Keep | Closes most of the gap left by a one-SIMD-group reduction. |
| Four-head RoPE reuse | Keep | Avoids repeated angle/sine/cosine work; current isolated latency is within roughly 7-14% of MLX. |
| Fused SwiGLU | Keep | One dispatch and no intermediates; approximately equal to MLX in isolated measurements. |
| 32-group online decode attention | Keep | Owns the complete attention algorithm; 8 and 16 groups exposed less parallel work on the course context. |
| No decode causal-mask graph | Keep | Pass `None` when `L = 1`; every cached position is already valid. |
| Normalize RoPE offsets once | Keep | Builds one batch offset array per model call rather than once per layer. |
| Last-token logits | Keep | Avoids vocabulary projection for prompt positions that generation never samples. |
| Fuse Q/K/V and gate/up projections | Reject | About 242 → 227 tok/s; fewer Python calls did not offset a more complex kernel and dispatch map. |
| Concatenate quantized weights at runtime | Reject | About 235 → 217 tok/s; constructing larger arrays added work and memory traffic. |
| Preallocate chunked dense KV cache | Reject | About 235 → 229 tok/s; strided logical slices hurt the following operations. Paging belongs in Week 3. |
| Share 32×16 prefill tiles in threadgroup memory | Reject | About 2,052 → 1,848 prefill tok/s; barriers outweighed reduced global loads at 128 tokens. |
| Broadcast scale/bias within the wide matvec | Reject | Increased vocabulary-head latency; extra loop structure and broadcasts outweighed parameter-load savings. |
| Four-output ordinary matvec | Reject | About 249 → 232 tok/s; extra register pressure outweighed activation reuse. |
| Affine identity in the ordinary matvec | Reject | About 249 → 244.5 tok/s; fewer arithmetic instructions did not improve the executed schedule. |
| Four SIMD groups per matvec threadgroup | Reject | Two runs near 248.5 tok/s; smaller threadgroups added scheduling overhead without useful new parallelism. |
| Sixteen groups for the vocabulary matvec | Reject | About 250.6 → 247.2 tok/s; a 512-thread scheduling unit offset the reduction in threadgroup count. |

This ledger is also a scheduling lesson. Fewer loads, fewer graph nodes, or
fewer dispatches are hypotheses. Only synchronized end-to-end measurement says
whether their occupancy, barrier, register, and layout tradeoffs helped.

## Expected Performance Contribution

**Estimated decode improvement: 5-15% beyond the operator kernels, with an
overall goal of at least 80% of matched MLX throughput.** Last-token logits and
graph cleanup avoid work outside the transformer blocks. Report the final
measured ratio; do not present this estimate as a substitute for it.

The current M1 Pro checkpoint uses Qwen3-0.6B-MLX-4bit, a 128-token prompt, 64
timed decode tokens, and two warmups. Recent stable course-owned runs are about
247-252 decode tok/s; the matched MLX run is about 317 tok/s. That is roughly
78-79% of MLX, about one percentage point below the 80% boundary. Prefill is
about 2,040-2,054 tok/s versus roughly 4,393 tok/s for MLX. This work-in-progress
result is reported honestly while the remaining quantized-projection gap is
optimized; the required path does not substitute MLX-provided operators to
manufacture a passing number.

{{#include copyright.md}}
