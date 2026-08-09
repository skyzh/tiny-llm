# 🚧 Week 2 Day 3: Quantize the Model

> **Status: Experimental.** See the
> [Week 2 verification matrix](./week2-overview.md#verification-status) for
> what is continuously tested, locally measured, and still under review.

Day 2 established a synchronized dense BF16 baseline. Day 3 reduces projection
weight traffic with packed W4A16 weights, implements the Metal operators that
consume them directly, wires those operators into the live model, and reruns
the same benchmark. Packed storage or an isolated fast kernel is not completion:
the cached model must use the quantized path.

**📚 Readings**

- [Model Compression and Quantization](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [MLX Extensions Development Guide](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [Quantized Matmul on GPU (Video)](https://www.youtube.com/watch?v=jYCxVirq4d0)

## Debug Metal Without a CPU Twin

A C++ CPU version is possible but not required. Use this three-level validation
ladder instead:

1. Write the equation in Python with `mlx.core`. This is the semantic oracle.
2. Translate it into a deliberately simple Metal kernel, usually with one
   thread responsible for one output element.
3. Optimize the validated Metal kernel with SIMD groups, vectorized loads, or
   SIMD-group matrix operations.

Compare each level with the one immediately above it. Do not debug an optimized
kernel by comparing only full-model text output.

### Make Failures Small and Synchronous

Start with deterministic fixtures whose expected values are easy to inspect:
zeros, ones, ramps, identity-like weights, and a fixed random seed. Exercise a
small aligned shape and then a tail shape. For example, test 8 and 10 rows for
an 8-row tile, or sequence lengths 32 and 35 for a 32-token block.

MLX execution is lazy, so force evaluation directly after the operator under
test. This turns a delayed compile or GPU execution failure into a failure at
the responsible call site:

```python
expected = python_reference(*inputs)
actual = metal_operator(*inputs)
mx.eval(expected, actual)

assert actual.shape == expected.shape
assert actual.dtype == mx.bfloat16
assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()
```

Check the wrapper boundary before inspecting the arithmetic. Assert the tensor
rank, shape, dtype, and contiguity assumptions in Python or C++, and verify that
the encoded buffer indices match the Metal function signature. Then classify
the failure:

- a pipeline creation error usually means the kernel name, specialization, or
  Metal compilation is wrong;
- an execution or address error usually means a grid, bounds check, stride, or
  buffer binding is wrong;
- a finite but inaccurate result usually means the indexing, reduction, mask,
  dequantization, or accumulator update is wrong.

For a numerical mismatch, temporarily simplify the schedule. Assign one output
to one thread, remove cooperative loads, and compare an intermediate such as a
dequantized weight group, a partial dot product, or an online-softmax row. A
small debug-only output buffer is often more useful than printing from every
GPU thread. Restore one optimization at a time and rerun both the aligned and
tail-shape tests after each change.

## Represent Weights With Fewer Bits

**Quantization** represents floating-point weights with values from a small
integer codebook plus the parameters needed to approximately reconstruct the
original values. This course uses **weight-only 4-bit quantization**:

- **W4** means that each logical weight is represented by a 4-bit code.
- **A16** means that activations and outputs remain 16-bit floating point.
- The resulting path is called **W4A16**. This course uses BF16 for its
  activations, scales, biases, and outputs.

With only 16 possible codes, the reconstructed weights approximate the original
values. The smaller representation trades some numerical precision for less
memory traffic.

The kernel does not materialize a dense BF16 weight matrix. It unpacks each
4-bit code, reconstructs the weight in registers, and immediately multiplies
it by the corresponding BF16 activation.

### Group-Wise Affine Quantization

Instead of applying one scale to an entire weight matrix, we divide each row
into **groups** and quantize every group independently. Local scales and biases
preserve more information about each group's weight distribution.

For a weight matrix $W$ of shape $(K, N)$, divide each row into groups of size
$G$. The Qwen3-4B MLX 4-bit checkpoint used in this course has a fixed group
size of 128:

```plain
Logical weight matrix W: K × N

Group size: G = 128
Number of groups per row = N / G

For each stored group of G consecutive values in a row:
  1. Unpack each unsigned 4-bit code q in [0, 15]
  2. Load the group's stored scale s and bias b
  3. Reconstruct each value as q * s + b
```

### Reconstruct a Stored Group

The checkpoint already contains the packed codes and their affine parameters.
For an unpacked unsigned code $q$, use the stored scale $s$ and bias $b$
directly:

$$
\hat{w} = q s + b
$$

The codes are unsigned, but the stored scale is signed. A positive scale maps
code 0 to the lower endpoint and code 15 toward the upper endpoint. A negative
scale reverses that orientation: code 0 is the upper endpoint and code 15 moves
toward the lower endpoint. Both orientations occur in the shipped Qwen3-4B MLX
checkpoint, so do not recompute `scale` and `bias` from an assumed min/max
orientation.

For example, these two stored parameter pairs reconstruct the same endpoint
range in opposite code order:

```plain
positive orientation: scale =  0.0867, bias = -0.5  => q=0 is -0.5, q=15 is about 0.8
negative orientation: scale = -0.0867, bias =  0.8  => q=0 is  0.8, q=15 is about -0.5
```

All required quantized-matmul tests use `group_size = 128` and BF16 scales,
biases, activations, and outputs. Normalize those tensors to BF16 in your
solution's model loader so every later kernel receives one model dtype.

### Packed Storage Layout

The 4-bit codes are packed for compact storage and efficient access:

```plain
Logical weight matrix: K × N
Dense BF16 storage: K × N bfloat16 (2 bytes each) = 2KN bytes
W4 code storage: K × N int4 (0.5 bytes each) = 0.5KN bytes

Packing: 8 × 4-bit values fit in one uint32 (32 bits)

Packed codes shape: K × (N / 8) uint32
Scales shape: K × (N / G) bfloat16
Biases shape: K × (N / G) bfloat16
```

Example packing for 8 consecutive 4-bit values `[a, b, c, d, e, f, g, h]`:

```plain
uint32_value = (h << 28) | (g << 24) | (f << 20) | (e << 16) |
               (d << 12) | (c << 8)  | (b << 4)  | a

Unpacking:
  a = (uint32_value >> 0)  & 0xF
  b = (uint32_value >> 4)  & 0xF
  c = (uint32_value >> 8)  & 0xF
  ...
  h = (uint32_value >> 28) & 0xF
```

## Revisit the Decode Roofline

The packed codes are not the entire W4 representation. Each group of 128
weights also stores one BF16 scale and one BF16 bias:

```plain
bytes per W4 weight = 0.5 + (2 + 2) / 128 = 0.53125 bytes
streamed W4 bytes   = 4,022,272,000 × 0.53125 = 2.137 GB per token
arithmetic intensity = 8.045 GFLOPs / 2.137 GB = 3.765 FLOPs/byte
```

Now W4 can be added to the dense comparison:

| Weight format | Value bits | Metadata per 128 weights | Effective bytes per weight | Streamed weight bytes per token | Weight arithmetic intensity |
|---|---:|---|---:|---:|---:|
| FP16 | 16 | None | 2 | 8.045 GB | 1.0 FLOP/byte |
| BF16 | 16 | None | 2 | 8.045 GB | 1.0 FLOP/byte |
| W4 | 4 | One BF16 scale and one BF16 bias | 0.53125 | 2.137 GB | 3.765 FLOPs/byte |

The smaller representation reduces the projection weight traffic by 3.765×.
That ratio is a bandwidth ceiling for one-token decode, not a promise of the
same end-to-end speedup.

### Theoretical Decode Roofline Across Apple Silicon

Apple publishes unified-memory bandwidth but not a directly comparable BF16
GPU TFLOPS figure. A bandwidth roofline can therefore be calculated without
assuming a compute ceiling:

```plain
ideal tokens/s = advertised memory bandwidth / streamed weight bytes per token
```

The table uses the highest-bandwidth configuration of each named chip. GB is
decimal, matching Apple's specifications. These are theoretical ceilings, not
benchmark results.

| Chip | Bandwidth | FP16/BF16 roofline | W4 roofline |
|---|---:|---:|---:|
| M1 Pro | 200 GB/s | 24.9 tok/s | 93.6 tok/s |
| M1 Max | 400 GB/s | 49.7 tok/s | 187.2 tok/s |
| M1 Ultra | 800 GB/s | 99.4 tok/s | 374.4 tok/s |
| M2 Pro | 200 GB/s | 24.9 tok/s | 93.6 tok/s |
| M2 Max | 400 GB/s | 49.7 tok/s | 187.2 tok/s |
| M2 Ultra | 800 GB/s | 99.4 tok/s | 374.4 tok/s |
| M3 Pro | 150 GB/s | 18.6 tok/s | 70.2 tok/s |
| M3 Max | 400 GB/s | 49.7 tok/s | 187.2 tok/s |
| M3 Ultra | 819 GB/s | 101.8 tok/s | 383.3 tok/s |
| M4 Pro | 273 GB/s | 33.9 tok/s | 127.8 tok/s |
| M4 Max | 546 GB/s | 67.9 tok/s | 255.5 tok/s |

The advertised bandwidths come from Apple's specifications for
[M1 Pro and Max](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/),
[M1 Ultra](https://www.apple.com/newsroom/2022/03/apple-unveils-m1-ultra-the-worlds-most-powerful-chip-for-a-personal-computer/),
[M2 Pro and Max](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/),
[M2 Ultra](https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/),
[M3 Pro and Max](https://support.apple.com/en-us/117736),
[M3 Ultra](https://www.apple.com/mac-studio/), and
[M4 Pro and Max](https://support.apple.com/en-us/121553). Apple's current Mac
Studio pairs M4 Max with M3 Ultra, so there is no M4 Ultra row.

These values assume peak advertised bandwidth, one read of every projection
weight, and no other traffic or work. Actual throughput is lower because the
complete model also reads activations and KV, launches other operators, and
does not sustain peak bandwidth continuously. The
[performance appendix](./appendix-performance.md) records measured results
separately from this theoretical exercise.

This roofline describes one-token decode, where `M = 1` and each streamed
weight serves one activation row. Prefill reuses each weight tile across many
rows, increasing arithmetic intensity. It therefore needs a matrix schedule;
the decode bandwidth ratio should not be treated as a prefill prediction.

## Quantized Matrix Multiplication

### Mathematical Formulation

For standard matrix multiplication $C = AB^T$ where:

- $A$: shape $(M, N)$, bfloat16 (activations)
- $B$: shape $(K, N)$, **quantized** to int4 (weights)
- $C$: shape $(M, K)$, same 16-bit dtype as $A$ (output)

Each element $C[i, k]$ is computed as:

$$
C[i, k] = \sum_{j=0}^{N-1} A[i, j] \times B[k, j]
$$

With quantization, $B[k, j]$ is represented as:

$$
B[k, j] = B_{\text{quantized}}[k, j] \times \text{scale}[k, g] + \text{bias}[k, g]
$$

where $g = \lfloor j / G \rfloor$ is the group index.

Substituting:

$$
C[i, k] = \sum_{g=0}^{N/G-1} \sum_{j'=0}^{G-1} A[i, g \times G + j'] \times (B_{\text{quantized}}[k, g \times G + j'] \times \text{scale}[k, g] + \text{bias}[k, g])
$$

Rearranging:

$$
C[i, k] = \sum_{g=0}^{N/G-1} \left( \text{scale}[k, g] \sum_{j'=0}^{G-1} A[i, g \times G + j'] \times B_{\text{quantized}}[k, g \times G + j'] + \text{bias}[k, g] \sum_{j'=0}^{G-1} A[i, g \times G + j'] \right)
$$

The scale and bias are constant within a group, so the computation can reuse
them across all values in that group.

### Computation Flow

```plain
Input:
  A: M × N (bfloat16 activations)
  B_quantized: K × (N/8) (uint32, packed weights)
  scales: K × (N/G) (bfloat16)
  biases: K × (N/G) (bfloat16)

Output:
  C: M × K (bfloat16)

For each output element C[i, k]:
  sum = 0  # float accumulator
  for each group g in 0..(N/G - 1):
    scale = scales[k, g]
    bias = biases[k, g]

    # Process G values in the group (G/8 uint32 packs)
    for each pack p in 0..(G/8 - 1):
      packed_value = B_quantized[k, g*(G/8) + p]

      # Unpack 8 × 4-bit values
      for bit_offset in [0, 4, 8, 12, 16, 20, 24, 28]:
        quantized = (packed_value >> bit_offset) & 0xF
        b_value = quantized * scale + bias
        a_value = A[i, g*G + p*8 + bit_offset/4]
        sum = sum + a_value * b_value

  C[i, k] = bfloat16(sum)
```

## Task 1: Implement Quantized Linear and Embedding

```
src/tiny_llm/quantize.py
src/tiny_llm/embedding.py
```

Modify these exact starter functions:

- `QuantizedWeights.from_mlx_layer`, `dequantize_weights`, and
  `quantized_linear` in `src/tiny_llm/quantize.py`;
- `QuantizedEmbedding.__call__` and `QuantizedEmbedding.as_linear` in
  `src/tiny_llm/embedding.py`.

The starter code provides `QuantizedWeights`, a container for a quantized
matrix and its dequantization parameters:

| Field | Shape | Description |
|-------|-------|-------------|
| `weight` | $(K, N/8)$ uint32 | Packed quantized weights. Each uint32 stores eight consecutive 4-bit values. |
| `scales` | $(K, N/G)$ bfloat16 | Stored signed per-group scale factors for dequantization. The sign determines which endpoint maps to the low codes. |
| `biases` | $(K, N/G)$ bfloat16 | Stored per-group offsets. Code 0 reconstructs to this value. |
| `group_size` | int | Number of consecutive values that share the same scale/bias. For the Qwen3 MLX 4-bit weights used here, this is `128`. |
| `bits` | int | Quantization bit width (typically 4, meaning values are in range $[0, 15]$) |

Its `from_mlx_layer` method extracts these fields from an MLX quantized layer
when loading the model.

Next, implement `quantized_linear`, a wrapper around `quantized_matmul` with the
same input convention as the standard `linear` function. You will implement
`quantized_matmul` in the next task.

Keep the token embedding table quantized as well. Add a `QuantizedEmbedding`
wrapper with two call patterns:

- `embedding(input_ids)` performs a row lookup. Gather the matching packed
  weights, scales, and biases. Unpack each `uint32` with shifts and masks,
  repeat each group's scale and bias across its 128 values, and compute
  `q * scale + bias` with basic `mlx.core` array operations. Do not call
  `mx.dequantize`. Put this unpacking logic in `dequantize_weights(...)` so
  the embedding and its direct tests share one explicit implementation.
- `embedding.as_linear(h)` is the tied output projection. Implement this with
  `quantized_linear(h, embedding_weight)` so it uses your quantized matmul path
  instead of materializing the full `vocab_size x hidden_size` table. This path
  starts working once the quantized matmul kernel is implemented in the next
  tasks.

## Task 2: Define the Quantized Matmul Primitive

```
src/extensions/src/tiny_llm_ext.h
src/extensions/bindings.cpp
src/extensions/src/quantized_matmul.cpp
src/extensions/CMakeLists.txt
```

The starter already contains the declaration, fail-closed source stub, binding,
and build registration. Keep the C++ declarations and definitions in the
`tiny_llm_ext` namespace and modify these exact functions:

- **`tiny_llm_ext.h`** — Read the Week 2 Day 3 `quantized_matmul(...)`
  declaration and `QuantizedMatmul` primitive interface; keep its signature in
  sync with the binding.
- **`bindings.cpp`** — Verify the existing `m.def("quantized_matmul", ...)`
  entry; do not create a second binding.
- **`quantized_matmul.cpp`** — Replace the body of
  `tiny_llm_ext::quantized_matmul(...)` to validate
  inputs, determine the output shape, return a lazy `mx::array`, and reject CPU
  evaluation explicitly in `QuantizedMatmul::eval_cpu(...)`.
- **`CMakeLists.txt`** — Verify the existing `quantized_matmul.cpp` source
  registration; do not add a duplicate.

The extension API is infrastructure: it lets an `mx.array` graph node schedule
the Metal loop you write in the next task. MLX owns the array lifetime and
command encoder, but it does not supply the quantized multiplication.

Build the extension to catch declaration, binding, and registration mismatches.
The focused test below checks the Task 1 Python wrappers; the primitive becomes
runnable after you implement its Metal schedules in Task 3:

```bash
pdm run build-ext
pdm run test --week 2 --day 3 -- -k task_1
```

## Task 3: Implement Metal Matrix Products

Before writing your first Metal kernel, understand the execution model. Metal
organizes GPU work in four nested scopes:

- **Lane (thread).** The smallest unit. Each lane executes the same
  instruction stream with its own register file. Lanes within a SIMD group
  can share data through `simd_` operations.
- **SIMD group (warp/subgroup).** A fixed-size set of lanes (32 on Apple
  GPUs) that execute in lockstep. `simd_sum`, `simd_shuffle`, and
  `simdgroup_matrix` operations work within this scope. A SIMD group cannot
  directly share registers with another SIMD group in the same threadgroup.
- **Threadgroup (block).** A collection of SIMD groups scheduled together on
  one GPU core. Threadgroups share threadgroup memory (explicitly allocated
  with `threadgroup` address space and synchronized with
  `threadgroup_barrier`). The grid is a 1D/2D/3D array of threadgroups.
- **Grid.** The total work dispatched. `dispatchThreadgroups` launches a grid
  of threadgroups; the GPU schedules them across available cores. Increasing
  the grid's threadgroup count can expose more independent work, but a finer
  partition can also duplicate reads or require partial-result merging.

Keep two launch knobs separate. More SIMD groups within one threadgroup add
threads and can raise register demand; they increase threadgroup-memory use
only when the schedule allocates shared storage per group or tile. Either
resource can reduce the number of resident threadgroups. More threadgroups in
the grid change how the output or reduction work is partitioned. Neither change
guarantees higher throughput.

Use the required two-SIMD-group matvec schedule as the Qwen starting point, then
benchmark two, four, eight, and sixteen groups per threadgroup as described below.
Change the grid partition separately so each measurement answers which launch
knob helped.

```
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

Modify these exact starter functions:

- `QuantizedMatmul::eval_gpu` in `quantized_matmul.cpp`;
- `quantized_matmul_vanilla_w4a16_g128` and
  `quantized_matvec_x4_fast_w4a16_g128` in `quantized_matmul.metal`;
- `quantized_matmul_vanilla` and `quantized_matvec_custom` in
  `src/tiny_llm/quantize.py` for the explicit comparison paths.

Write the Metal kernels and connect `eval_gpu` to them. The Python
`quantized_matmul` wrapper always dispatches the primitive you implement on
GPU; the required path in your solution never routes through
`mx.quantized_matmul`.

Do this in two measured stages. They expose the same math but schedule
different shapes differently:

1. **Vanilla matmul:** one Metal thread computes one output element. This is
   the direct GPU translation of the computation flow above and an inspectable
   bring-up control.
2. **SIMD matvec:** for decode, SIMD lanes cooperate on the reduction for one
   activation row and calculate several output columns together.

Here, `M` is the number of activation rows after flattening every leading
dimension. Day 3 uses this explicit dispatch:

| Activation rows | Kernel | Role at this checkpoint |
|---:|---|---|
| `M <= 8` | SIMD matvec | Optimized path for decode and other very small matrix inputs. |
| `M > 8` | Vanilla matmul | Correctness-first prefill path; Day 6 replaces it with a cooperative tiled kernel. |

The cutoff does not mean the SIMD kernel expands to cover larger `M`. The two
paths are separate schedules: Day 3 optimizes the vector-shaped decode
bottleneck and leaves matrix-shaped prefill visible for the later benchmark to
select.

Keep the vanilla function callable as `quantized_matmul_vanilla`. An
optimization is much easier to trust when it can be compared directly with
the implementation it replaces.

### Stage 1: Vanilla Matmul

Start with a two-dimensional grid over output row `i` and output column `k`.
Each thread walks all `N` input values, unpacks eight int4 weights from each
`uint32`, applies the group scale and bias, and accumulates one `C[i, k]` in
float32. This kernel repeats activation loads and does not share work, but its
control flow mirrors the equation and makes it a useful debugging control. The
Python `mlx.core` equation remains the correctness oracle for both Metal
schedules.

Keep the vanilla kernel for matrix-shaped prefill in this chapter; Day 6
revisits that workload with cooperative tiling.

### Stage 2: SIMD Matvec

Decode normally has `M = 1`; an 8×8 matrix tile would leave most rows empty.
Instead, one SIMD group reduces the input dimension and uses `simd_sum` to
combine lane-local partial sums. Start with two output columns per group as an
inspectable schedule. For the Qwen3-4B checkpoint, then evaluate a four-column
path in which each lane loads two adjacent packed words, or 16 activations, and
reuses them across the four outputs.

The optimized path also uses the affine identity

$$
\sum_j a_j(sq_j+b) = s\sum_j a_jq_j + b\sum_j a_j
$$

to avoid applying the bias separately to every unpacked value. It also scales
the activations once and reads four packed int4 values through a 16-bit mask,
avoiding a shift for every weight and output row. This adds live accumulators,
so test it as a complete schedule rather than assuming fewer integer
instructions must be faster.

### Tune the SIMD Schedule

Treat output width, threadgroup size, and shared-memory reuse as benchmark
variables. Use this Qwen-focused starting point:

- flatten all leading activation dimensions into `M`,
- use the custom matvec when `M <= 8` and the vanilla matmul when `M > 8`,
- compute four output columns per SIMD group and load two adjacent packed words
  per lane,
- launch two SIMD groups, or eight output rows, per threadgroup.

These thresholds are measured starting points, not mathematical requirements.
Keep them visible in the dispatcher, then vary one choice at a time. Compare
two, four, and eight output columns per SIMD group. More columns increase
activation reuse, but also extend accumulator lifetimes and raise register
pressure. Compare two, four, eight, and sixteen SIMD groups per threadgroup.
More groups expose additional outputs, but may duplicate activation reads and
reduce residency.

Evaluate the affine rearrangement as part of the complete schedule. Its lower
instruction count is useful only if the longer-lived activation sum and output
accumulators do not reduce occupancy. Select the schedule with a synchronized
whole-model decode benchmark, not an instruction-count estimate.

Define a row-contiguous Python-to-extension contract for scales, biases,
activations, and packed weights. Call `mx.contiguous` once at that boundary and
validate the layout in the C++ primitive before encoding the kernel. Metal
receives raw buffers rather than implicit array strides, so layout is a
correctness condition as well as a performance condition.

Use direct activation reads for your kernel. The one-row activation is
small and cache-friendly, while staging it in threadgroup memory adds a barrier
to every projection. If you test shared staging as an ablation, report the
whole-model result and keep it only when reuse outweighs synchronization.

### Kernel Requirements

Implement both required kernel layouts in `quantized_matmul.metal`:

- First, implement the vanilla one-thread-per-output matrix grid.
- For `M <= 8`, assign one SIMD group to an output tile. Cooperatively reduce
  the input dimension and compute several output columns per group.
- For `M > 8`, dispatch the vanilla matrix grid. Do not loop over rows with the
  SIMD matvec schedule; Day 6 introduces the tiled prefill schedule.
- The required kernel supports `bfloat16_t` inputs and outputs. The Week 2
  checkpoint does not add a second model-storage dtype.
- Apply the group-wise dequantization loop defined earlier in this chapter:
  - Iterate over groups of 128 values.
  - Unpack int4 values from each `uint32`.
  - Dequantize each value with `q * scale + bias`.
  - Accumulate products in `float`, then cast the result to the kernel dtype.
- Add boundary checks (`i < M`, `k < K`) before writing output.

The custom kernel only needs to support `bits = 4` and `group_size = 128`. Use
the group size to compute `groups_per_row` and the packed-weight offsets.
Instantiate the required Metal kernel for `bfloat16_t` and select it in
`eval_gpu`. If you retain an optional `half` specialization, keep it out of the
model dispatch in your solution.

### GPU Dispatch

Complete `eval_gpu` in `quantized_matmul.cpp` by following `axpby`'s GPU
dispatch pattern:

1. Get the Metal device and command encoder from the stream.
2. Load the quantized matmul kernel matching the output dtype from the Metal
   library.
3. Bind the input and output buffers and the dimension constants (`M`, `N`,
   `K`). The buffer order must match the kernel signature.
4. Select the matrix-vector layout for `M <= 8`; otherwise select the vanilla
   matrix layout. Keep both paths explicit for direct comparisons.
   Calculate a SIMD-aligned thread-group configuration and tile output columns
   so packed input values and activations can be reused. Use the four-column,
   two-packed-word kernel with two SIMD groups.
5. Dispatch with `dispatchThreadgroups`.

You can test your solution by running:

```bash
pdm run build-ext
pdm run test --week 2 --day 3 -- -k gpu
```

The direct tests cover matvec at `M = 1` and `M = 8`, the vanilla matmul at
`M = 128`, and compare them with an MLX oracle. The oracle checks the result;
it is not the implementation under test.

## Task 4: Integrate Before Continuing

```
src/tiny_llm/qwen3_week2.py
```

Modify `Qwen3ModelWeek2.__init__`, `Qwen3MultiHeadAttention.__call__`,
`Qwen3MLP.__call__`, and `Qwen3ModelWeek2.__call__` in this task. These are the
exact points that load quantized weights, replace dense projections, and keep
only the requested logits row.

Integrate quantized matrix multiplication into the Week 2 Qwen3 model so that
the linear layers remain quantized throughout inference.

Change the weight type from `mx.array` to `QuantizedWeights` for every
attention projection (`wq`, `wk`, `wv`, and `wo`) and MLP projection (`w_gate`,
`w_up`, and `w_down`). Replace `linear(x, w)` with `quantized_linear(x, w)`. In
the Week 2 model loader, use `QuantizedWeights.from_mlx_layer(...)` instead of
materializing a 16-bit matrix. Keep the Week 1 model's boundary intact; its
layers still expect plain `mx.array` weights.

For embeddings, wire the `QuantizedEmbedding` from Task 1 into the loader: load
`embed_tokens` with `QuantizedWeights.from_mlx_layer(...)` and pass it to
`QuantizedEmbedding`. If the model has a separate `lm_head`, keep that head as
`QuantizedWeights` too and apply it with `quantized_linear`; `lm_head` is a
projection, not an embedding lookup.

Normalize each loaded layer's scales and biases to BF16. Require scales,
biases, and activations to match and return BF16. If the output is `nan` or
otherwise invalid, check for a dtype mismatch first.

Preserve the quantized layer's parameters as well. The model should pass
`w.group_size` and `w.bits` to the extension, which should validate the course
assumptions: `group_size = 128` and `bits = 4`.

You can test your solution by running:

```bash
pdm run test --week 2 --day 3

pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-4b
```

You can also benchmark your solution:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

Run the same command with `--solution tiny_llm_ref` to compare it with the
reference solution.

The vanilla matrix product remains callable as an inspectable Metal control,
but the Python `mlx.core` equation is the correctness oracle and only the SIMD
matvec is integrated into decode.

## Verify Quantization in the Complete Model

Before moving on, confirm that the quantized matvec kernel is actually called
during model inference, not just registered and tested in isolation.

> **🚧 Acceptance criterion.** Your checkpoint is incomplete until the model's
> projection dispatcher is wired to your custom primitive. Decode-shaped work
> must route through `quantized_linear` → `quantized_matvec_custom` → the
> extension primitive → the Metal matvec. Matrix-shaped work must route through
> `quantized_linear` → `quantized_matmul` → the extension primitive → its Metal
> matrix schedule. Use a source trace through those branches in your completed
> dispatcher and model wiring. The supplied tests validate packed model state
> and the direct operators, while the matched benchmark reports complete-model
> throughput; neither proves the live Metal pipeline identity by itself. Use a
> direct source trace of the dispatch branches, then treat the throughput
> comparison as a separate result.

Measure the cumulative model and the real projection shapes:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-kv-cache --variant week2-quantized-matvec --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last

pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section decode-projections --context 128
```

Attach the complete-model before/after rows, the per-projection latency table,
and the direct dispatch trace. First require a clear decode gain over
`kv-cache`. Then compare each projection with MLX at the identical shape.
Projections may remain the largest absolute category because the model performs
them in every layer; once their operator latency is close to MLX, that bar is
no longer the largest removable gap.

Continue to Day 4 only after the correctness tests pass, the source trace proves
that the live model selects the intended matrix and matvec branches, the matched
model run improves decode over `kv-cache`, and the projection table is close to
MLX at the same shapes. If the projection comparison is still far behind, keep
tuning the matvec instead. Once that gap shrinks, Day 4 turns to the recurring
normalization, position, and activation work around those projections.

> **Optional profiling evidence.** A kernel-group replay or operator attribution
> can corroborate that transition, but neither gates progress. The
> [reference checkpoint](./appendix-performance.md#day-3-keep-weights-packed)
> includes both alongside the model and projection measurements above.

{{#include copyright.md}}
