# 🚧 Week 2 Day 6: SIMD-Matrix Prefill

> 🚧 This chapter is under review and may change.

Day 3 optimized the one-to-eight-row matrix-vector shape used during decode.
Prefill is a different workload: it projects tens or hundreds of token rows at
once. Today we add a second quantized-linear schedule for that larger `M`
dimension and introduce Metal's 8×8 SIMD-matrix fragments. Week 3 will reuse
the same fragment API after paged KV addressing is established.

This chapter assumes the packed W4A16 format and dispatch interface from Day 3,
the synchronized prefill benchmark from Day 2, and the BF16 model contract from
the Week 2 overview. No quantization format changes today.

## From SIMD Reduction to SIMD-Matrix Fragments

The Day 3 decode kernel gives a SIMD group several output columns and reduces
each dot product across its 32 lanes. That schedule fits a very small number of
activation rows. During prefill, the same weight row is reused across many
activation rows, so it is better to tile both output dimensions.

Metal exposes an 8×8 `simdgroup_matrix` fragment. A fragment is a register-held
matrix tile distributed across the 32 lanes of one SIMD group; it is not an
ordinary thread-local array. The public matrix-multiply-accumulate operation
computes:

```plain
C[8, 8] += A[8, 8] × B[8, 8]
```

Each SIMD group owns one 8×8 output tile. It advances through the reduction
dimension eight values at a time, loading one activation fragment and one
dequantized weight fragment before issuing the multiply-accumulate.

```plain
activation rows M
        |
        v
  +-----------+     reduction N in 8-value steps
  | A 8 x 8   |  ×  | B 8 x 8 |  --->  C 8 x 8
  +-----------+                         one SIMD group
        |
        +---- next 8 activation rows

output columns K are covered by independent 8-column tiles
```

This is the first use of SIMD-matrix fragments in the course. Keep the scalar
Day 3 kernel as the correctness oracle while bringing up the tiled path.

## Mixed-Precision Boundary

The completed course model stores activations, scales, biases, and outputs in
BF16. Unpack each 4-bit weight and combine it with its BF16 scale and bias, but
accumulate the matrix product in FP32 fragments. Cast only once when storing the
final BF16 output tile.

This distinction applies for the rest of the course:

| Location | Dtype |
| --- | --- |
| Model activations and KV storage | BF16 |
| Quantization scale and bias storage | BF16 |
| Matrix operands loaded by the kernel | BF16 |
| Dot-product and online-softmax accumulators | FP32 |
| Model-facing kernel output | BF16 |

BF16 is the model-storage contract; FP32 is an internal arithmetic choice. A
correct result with an FP32 output still violates the model contract.

## Task 1: Add Shape-Based Dispatch

Keep the Day 3 matrix-vector path for `M <= 8`. For larger `M`, dispatch a new
SIMD-matrix kernel:

```plain
M <= 8  -> quantized SIMD matvec
M > 8   -> quantized SIMD-matrix matmul
```

Expose the prefill path through `QuantizedWeights.use_simdgroup_matmul` while
you compare checkpoints. The dispatch must preserve the same function
signature, packed weights, output shape, and BF16 dtype as Day 3.

## Task 2: Implement the 8×8 Kernel

Use one SIMD group per 8×8 output tile. For every eight-wide reduction step:

1. Load the activation fragment cooperatively.
2. Unpack the matching 4-bit weights.
3. Apply the group scale and bias while forming the weight fragment.
4. Issue `simdgroup_multiply_accumulate` into an FP32 accumulator fragment.
5. After the complete reduction, store only valid rows and columns as BF16.

Zero-fill partial input fragments. Guard partial output tiles at the final
store. A prefill length such as ten tokens must exercise both a complete tile
and a two-row tail without changing the accumulation dtype.

## Task 3: Hoist Quantization Parameters

One scale and bias cover 128 reduction elements, or sixteen consecutive
eight-wide matrix steps. Loading them inside every step repeats address
calculation and device reads.

Loop over quantization groups first. Load the scale and bias values needed by
the output fragment into registers, then reuse them for all sixteen steps:

```plain
for each 128-value quantization group:
    load scale and bias for this output fragment
    for each of its sixteen 8-value reduction tiles:
        unpack and dequantize weights
        matrix-multiply-accumulate into FP32
```

Measure this change independently. More reuse is only a hypothesis until a
synchronized benchmark shows that its register cost is worthwhile.

## Task 4: Fuse Quantized Embedding Lookup

Prompt tokens do not need a matrix multiplication. They select rows from the
quantized embedding table and dequantize only those rows. The readable Day 3
path expresses gather, int4 unpacking, scale, and bias as separate array
operations. Add a direct Metal kernel that fuses those steps into one dispatch.

Map one thread to one requested embedding element. Accept both int32 prompt IDs
and uint32 sampled IDs so generation does not insert a token-cast graph node.
This kernel reuses the W4A16 unpacking equation from Day 3; the only new idea is
fusing row selection with dequantization.

## Task 5: Keep Only Needed Logits

Prefill computes hidden states for the whole prompt, but generation samples only
from the final position. Preserve an optional `logits_to_keep` model argument
and slice hidden states before the final norm and vocabulary projection:

```python
if logits_to_keep is not None:
    h = h[:, -logits_to_keep:, :]
```

Generation requests one row. Correctness tests and prompt-scoring callers can
still pass `None` to obtain every position. This is a model-interface
optimization, not a change to attention or sampling.

## Task 6: Integrate the Prefill Checkpoint

Add a cumulative `simd-matmul` checkpoint after `swiglu`. It enables
`use_simdgroup_matmul` for every quantized projection and the direct embedding
kernel while retaining the Day 3 matvec for decode. Update generation to request
only the final logit row. Run:

```bash
pdm run test --week 2 --day 6

pdm run bench-week2-progression --offline --repeats 3 \
  --model qwen3-0.6b --input-len 128 --output-len 65 --warmup 2
```

Report prefill and decode separately. The new schedule should improve prefill;
it should not change the one-token decode path.

## Prepare for Paged FlashAttention

Day 4 introduced online softmax, and this chapter introduced BF16
SIMD-matrix fragments. Week 3 first adds page-table translation, then combines
these established ideas in paged FlashAttention:
one SIMD-matrix product forms score tiles, online softmax updates the running
state, and a second SIMD-matrix product applies probabilities to value tiles.

{{#include copyright.md}}
