# Week 2 Day 5: Decode Attention

During single-request decode, the query length is one while the cached key/value
sequence grows. The readable grouped-query implementation materializes scores
and repeats shape work that MLX's optimized attention primitive handles more
efficiently.

## Task 1: Add the Optimized Interface

Implement `scaled_dot_product_attention` in `week2_kernels.py` with
`mx.fast.scaled_dot_product_attention`. Preserve the model-facing shapes:

```plain
query: B, H_q, L, D
key:   B, H_kv, S, D
value: B, H_kv, S, D
out:   B, H_q, L, D
```

Pass the scale and mask through directly. MLX handles grouped-query attention
when `H_q` is a multiple of `H_kv`.

## Task 2: Use It for Decode

Route the Week 2 attention module through the optimized interface. Keep the
FlashAttention implementation out of this path: its tiled prefill workload and
serving integration belong to Week 3.

```bash
pdm run test --week 2 --day 5
```

The test compares output shape and values with the readable grouped-query
attention implementation using different query and KV head counts.

{{#include copyright.md}}
