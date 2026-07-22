# Week 2 Day 4: Fast RMSNorm, RoPE, and SwiGLU

The Week 1 operators favor readable equations. Week 2 adds optimized versions
behind a dedicated module instead of rewriting those implementations.

```plain
src/tiny_llm/week2_kernels.py
src/tiny_llm_ref/week2_kernels.py
```

## Task 1: Fast RMSNorm

Implement `FastRMSNorm` with `mx.fast.rms_norm`. Preserve the constructor and
call shape of the readable `RMSNorm` so the model can select the implementation
without changing its surrounding logic.

## Task 2: Fast RoPE

Implement `FastRoPE` with `mx.fast.rope`. The Qwen model stores tensors as
`B, L, H, D`, while the primitive expects `B, H, L, D`, so transpose on entry
and restore the original layout on return. Support both one scalar offset and
one offset per batch element.

## Task 3: SwiGLU

Express SwiGLU as one array expression:

```python
gate * mx.sigmoid(gate) * up
```

Keeping it in one expression allows MLX to fuse the elementwise work. Do not
change the Week 1 `silu` implementation.

## Task 4: Integrate the Layer

Update `qwen3_week2.py` to import the Week 2 operations. `qwen3_week3.py` must
reuse the same interfaces so serving features do not regress to the readable
Week 1 kernels.

```bash
pdm run test --week 2 --day 4
```

The tests compare the fast operations with the readable versions, exercise
scalar and per-batch RoPE offsets, and verify the Week 1/2/3 boundaries.

{{#include copyright.md}}
