# 🚧 Week 2 Day 3: Quantized Matvec

> 🚧 This chapter is under review and may change.

Day 2's decode profile should show the same scaling before you start this
chapter: linear projections repeatedly read far more weight data than one-token
activations. We therefore study and implement quantized matrix multiplication.
Quantizing weights from 16-bit floating point to 4-bit integers reduces both
model size and the memory traffic required for each generated token.

**📚 Readings**

- [Model Compression and Quantization](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [MLX Extensions Development Guide](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [Quantized Matmul on GPU (Video)](https://www.youtube.com/watch?v=jYCxVirq4d0)

## Why Quantization?

The decode phase of LLM inference is typically **memory-bandwidth bound**: each
token requires reading the model's weights but performs relatively little work
with them. Use the dimensions in the official
[Qwen3-4B configuration](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json)
to calculate the ideal bound:

```plain
Qwen3-4B dimensions:
  hidden size        h = 2,560
  MLP size           i = 9,728
  query width        q = 4,096
  key/value width   kv = 1,024
  layers             L = 36
  vocabulary         V = 151,936

Projection weights per layer:
  Q and O: 2 × h × q       =  20,971,520
  K and V: 2 × h × kv      =   5,242,880
  MLP:     3 × h × i       =  74,711,040
  total per layer          = 100,925,440

All transformer layers: L × 100,925,440 = 3,633,315,840
Tied vocabulary head:    V × h           =   388,956,160
Total streamed weights:                    4,022,272,000

FLOPs per token: 2 × 4,022,272,000 = 8.045 GFLOPs
```

The tied embedding matrix is counted once as the vocabulary projection. The
single-row embedding lookup, normalization weights, activations, KV reads, and
attention work are omitted. This makes the result an upper bound for linear
layers, not a prediction of complete-model throughput.

For BF16, the streamed weights occupy:

```plain
4,022,272,000 weights × 2 bytes = 8.045 GB per token
arithmetic intensity = 8.045 GFLOPs / 8.045 GB = 1.0 FLOP/byte
```

The course uses affine W4 groups of 128 values. Packed weights need 0.5 byte
per value, and each group stores one BF16 scale plus one BF16 bias:

```plain
bytes per W4 weight = 0.5 + (2 + 2) / 128 = 0.53125 bytes
streamed W4 bytes   = 4,022,272,000 × 0.53125 = 2.137 GB per token
arithmetic intensity = 8.045 GFLOPs / 2.137 GB = 3.765 FLOPs/byte
```

### Decode Roofline Across Apple Silicon

Apple publishes unified-memory bandwidth but not a directly comparable BF16
GPU TFLOPS figure. We can therefore calculate a bandwidth roofline without
inventing a compute ceiling:

```plain
ideal tokens/s = advertised memory bandwidth / streamed weight bytes per token
```

The table uses the highest-bandwidth configuration of each named chip. GB is
decimal, matching Apple's specifications.

| Chip | Bandwidth | BF16 roofline | W4 roofline | Ideal W4 work rate |
|---|---:|---:|---:|---:|
| M1 Pro | 200 GB/s | 24.9 tok/s | 93.6 tok/s | 0.75 TFLOPS |
| M1 Max | 400 GB/s | 49.7 tok/s | 187.2 tok/s | 1.51 TFLOPS |
| M1 Ultra | 800 GB/s | 99.4 tok/s | 374.4 tok/s | 3.01 TFLOPS |
| M2 Pro | 200 GB/s | 24.9 tok/s | 93.6 tok/s | 0.75 TFLOPS |
| M2 Max | 400 GB/s | 49.7 tok/s | 187.2 tok/s | 1.51 TFLOPS |
| M2 Ultra | 800 GB/s | 99.4 tok/s | 374.4 tok/s | 3.01 TFLOPS |
| M3 Pro | 150 GB/s | 18.6 tok/s | 70.2 tok/s | 0.56 TFLOPS |
| M3 Max | 400 GB/s | 49.7 tok/s | 187.2 tok/s | 1.51 TFLOPS |
| M3 Ultra | 819 GB/s | 101.8 tok/s | 383.3 tok/s | 3.08 TFLOPS |
| M4 Pro | 273 GB/s | 33.9 tok/s | 127.8 tok/s | 1.03 TFLOPS |
| M4 Max | 546 GB/s | 67.9 tok/s | 255.5 tok/s | 2.06 TFLOPS |

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
weight, and no other traffic or work. Actual throughput is lower. On the
reference M4 Pro, MLX reaches 87.58 decode tok/s and the completed course model
reaches 77.41 tok/s, compared with the ideal W4 roofline of 127.8 tok/s.

### Why This Does Not Predict a 4× Prefill Gain

The decode calculation has `M = 1`: each streamed weight serves one activation
row. Prefill reuses a weight tile across many rows. Under the same one-read
idealization, weight arithmetic intensity scales with `M`. The Day 6 tile
guarantees reuse across 32 rows even when separate row tiles receive no cache
reuse:

| Workload | `M` | BF16 weight intensity | W4 weight intensity | Likely limit |
|---|---:|---:|---:|---|
| Decode | 1 | 1.0 FLOP/byte | 3.765 FLOPs/byte | Weight bandwidth |
| One Day 6 row tile | 32 | 32 FLOPs/byte | 120.5 FLOPs/byte | Matrix schedule and dequantization |
| Full Week 2 prefill ideal | 128 | 128 FLOPs/byte | 482 FLOPs/byte | Matrix schedule and dequantization |

Day 2's dense BF16 prefill is therefore already compute-oriented. Day 6's goal
is to keep weights packed while recovering that efficient matrix throughput,
not to multiply prefill tokens per second by four. The measured progression is
726.25 tok/s on Day 2 and 792.18 tok/s after Day 7, a 9.1% gain.

### The Solution: Quantization

Compressing BF16 weights to
4-bit integers (int4) can:

- **Reduce decode weight traffic by 3.765×**: 8.045 GB → 2.137 GB per token
- **Improve decode weight arithmetic intensity by 3.765×**: 1.0 → 3.765 FLOPs/byte
- **Raise the bandwidth roofline for decode projections by up to 3.765×**

The tradeoff is a small loss in model accuracy when the weights are quantized
carefully.

### Group-wise Quantization

Instead of applying one scale to an entire weight matrix, we divide each row
into **groups** and quantize every group independently. Local scales and biases
preserve more information about each group's weight distribution.

For a weight matrix $W$ of shape $(K, N)$, divide each row into groups of size
$G$. The Qwen3-4B MLX 4-bit checkpoint used in this course has a fixed group
size of 128:

```plain
Original weight matrix W: K × N (bfloat16)

Group size: G = 128
Number of groups per row = N / G

For each group of G consecutive values in a row:
  1. Find min and max values
  2. Compute scale and bias to map [min, max] → [0, 15] (4-bit range)
  3. Quantize each value using: quantized = round((value - bias) / scale)
```

All required quantized-matmul tests use `group_size = 128` and BF16 scales,
biases, activations, and outputs. Normalize those tensors to BF16 when loading
the course model so every later kernel receives one model dtype.

### Affine Quantization

We use **affine (asymmetric) quantization**, which maps a floating-point range
onto the full integer range:

$$
\text{quantized} = \text{round}\left(\frac{\text{value} - \text{bias}}{\text{scale}}\right)
$$

$$
\text{dequantized} = \text{quantized} \times \text{scale} + \text{bias}
$$

For 4-bit quantization, the quantized values are in the range $[0, 15]$.

Given a group with minimum value $v_{min}$ and maximum value $v_{max}$:

$$
\text{scale} = \frac{v_{max} - v_{min}}{2^{\text{bits}} - 1} = \frac{v_{max} - v_{min}}{15}
$$

$$
\text{bias} = v_{min}
$$

**Example:**

```plain
Group values: [-0.5, -0.3, 0.1, 0.4, 0.8]
min = -0.5, max = 0.8

scale = (0.8 - (-0.5)) / 15 = 1.3 / 15 ≈ 0.0867
bias = -0.5

Quantization:
  -0.5 → round((-0.5 - (-0.5)) / 0.0867) = 0
  -0.3 → round((-0.3 - (-0.5)) / 0.0867) = 2
   0.1 → round((0.1 - (-0.5)) / 0.0867) = 7
   0.4 → round((0.4 - (-0.5)) / 0.0867) = 10
   0.8 → round((0.8 - (-0.5)) / 0.0867) = 15

Quantized: [0, 2, 7, 10, 15] (4 bits each)
```

### Storage Format

The quantized values are packed for compact storage and efficient access:

```plain
Original: K × N bfloat16 (2 bytes each) = 2KN bytes
Quantized: K × N int4 (0.5 bytes each) = 0.5KN bytes

Packing: 8 × 4-bit values fit in one uint32 (32 bits)

Weight matrix shape: K × N
Quantized storage shape: K × (N / 8) uint32
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

The starter code provides `QuantizedWeights`, a container for a quantized
matrix and its dequantization parameters:

| Field | Shape | Description |
|-------|-------|-------------|
| `weight` | $(K, N/8)$ uint32 | Packed quantized weights. Each uint32 stores eight consecutive 4-bit values. |
| `scales` | $(K, N/G)$ bfloat16 | Per-group scale factors for dequantization. Each group of $G$ consecutive values shares one scale. Recall: $\text{scale} = (v_{max} - v_{min}) / 15$ |
| `biases` | $(K, N/G)$ bfloat16 | Per-group bias (offset) for dequantization. Recall: $\text{bias} = v_{min}$ |
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
  `mx.dequantize`. Put this readable unpacking logic in
  `dequantize_weights(...)` so the embedding and its direct tests share one
  explicit implementation.
- `embedding.as_linear(h)` is the tied output projection. Implement this with
  `quantized_linear(h, embedding_weight)` so it uses your quantized matmul path
  instead of materializing the full `vocab_size x hidden_size` table. This path
  starts working once the quantized matmul kernel is implemented in the next
  tasks.

## Task 2: Register a GPU-Only Primitive

Register quantized matrix multiplication as an MLX C++ extension. Follow the
existing `axpby` example for array validation, lazy primitive construction,
bindings, and Metal dispatch. The course implementation is GPU-only; its
`eval_cpu` method should raise a clear unsupported-device error.

```
src/extensions/src/tiny_llm_ext.h
src/extensions/bindings.cpp
src/extensions/src/quantized_matmul.cpp
src/extensions/CMakeLists.txt
```

You will update four files. Keep the C++ declarations and definitions in the
`tiny_llm_ext` namespace:

- **`tiny_llm_ext.h`** — Declare the `quantized_matmul(...)` function signature and define a `QuantizedMatmul` primitive class (inheriting `mx::Primitive`). Store `group_size` and `bits` as private members.
- **`bindings.cpp`** — Add an `m.def(...)` call to expose the function to Python.
- **`quantized_matmul.cpp`** — Implement `quantized_matmul(...)` to validate
  inputs, determine the output shape, return a lazy `mx::array`, and reject CPU
  evaluation explicitly.
- **`CMakeLists.txt`** — Add the new C++ source to the extension target.

The extension API is infrastructure: it lets an `mx.array` graph node schedule
the Metal loop you write in the next task. MLX owns the array lifetime and
command encoder, but it does not supply the quantized multiplication.

Build and test the extension:

```bash
pdm run build-ext
pdm run test --week 2 --day 3 -- -k task_1
```

## Task 3: Implement Metal Matrix Products

```
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

Write the Metal kernels and connect `eval_gpu` to them. The Python
`quantized_matmul` wrapper always dispatches this course-owned primitive on
GPU; the required path never routes through `mx.quantized_matmul`.

Do this in two measured stages. They expose the same math but schedule
different shapes differently:

1. **Vanilla matmul:** one Metal thread computes one output element. This is
   the direct GPU translation of the computation flow above and the correctness
   baseline.
2. **SIMD matvec:** for decode, SIMD lanes cooperate on the reduction for one
   activation row and calculate several output columns together.

Keep the vanilla function callable as `quantized_matmul_vanilla`. An
optimization is much easier to trust when it can be compared directly with
the implementation it replaces.

### Stage 1: Vanilla Matmul

Start with a two-dimensional grid over output row `i` and output column `k`.
Each thread walks all `N` input values, unpacks eight int4 weights from each
`uint32`, applies the group scale and bias, and accumulates one `C[i, k]` in
float32. This kernel repeats activation loads and does not share work, but its
control flow mirrors the equation and is a useful debugging oracle.

### Prefill Tiling Comes on Day 6

Prefill has many activation rows and benefits from a different matrix-matrix
schedule. Keep the vanilla kernel for that shape today so Day 3 stays focused
on decode. Day 6 replaces it with a cooperative 32×32 tile built from 8×8
`simdgroup_matrix` fragments after the course has established the packed
format, dispatcher, and synchronized benchmark.

### Stage 2: SIMD Matvec

Decode normally has `M = 1`; an 8×8 matrix tile would leave most rows empty.
Instead, one SIMD group reduces the input dimension and uses `simd_sum` to
combine lane-local partial sums. Start with two output columns per group as an
inspectable schedule. The Qwen3-4B profile then motivates a faster
path: each lane loads two adjacent packed words, or 16 activations, and reuses
them across four output columns.

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
- use the custom matvec when `M <= 8`,
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

At the Python-to-extension boundary, make scales, biases, activations, and
packed weights row-contiguous once with `mx.contiguous`. The C++ primitive
validates that contract before encoding the kernel. A Metal kernel receives
raw buffers and strides are not implicit, so silently accepting a noncontiguous
view would produce either wrong addressing or a slower hidden copy in a less
explicit layer.

Use direct activation reads for the course kernel. The one-row activation is
small and cache-friendly, while staging it in threadgroup memory adds a barrier
to every projection. If you test shared staging as an ablation, report the
whole-model result and keep it only when reuse outweighs synchronization.

### Direct Quantized Embedding Comes on Day 6

Week 2 performs row lookup and dequantization with basic `mlx.core` array
operations at this checkpoint. Day 6 optionally fuses row gather, int4
unpacking, and affine dequantization into one Metal dispatch.

### Kernel Requirements

Implement both required kernel layouts in `quantized_matmul.metal`:

- First, implement the vanilla one-thread-per-output matrix grid.
- For `M <= 8`, assign one SIMD group to an output tile. Cooperatively reduce
  the input dimension and compute several output columns per group.
- The required kernel supports `bfloat16_t` inputs and outputs. The course
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
course-model dispatch.

### GPU Dispatch

Complete `eval_gpu` in `quantized_matmul.cpp` by following `axpby`'s GPU dispatch
pattern:

1. Get the Metal device and command encoder from the stream.
2. Load the quantized matmul kernel matching the output dtype from the Metal library.
3. Bind the input and output buffers and the dimension constants (`M`, `N`,
   `K`). The buffer order must match the kernel signature.
4. Select the matrix-vector layout for `M <= 8`; otherwise select the vanilla
   matrix layout. Keep both paths explicit for direct comparisons.
   Calculate a SIMD-aligned thread-group configuration and tile output columns
   so packed input values and activations can be reused. Use the four-column,
   two-packed-word kernel with two SIMD groups.
5. Dispatch with `dispatchThreadgroups`.

You can test your implementation by running:

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

Integrate quantized matrix multiplication into the Week 2 Qwen3 model so that
the linear layers remain quantized throughout inference.

Change the weight type from `mx.array` to `QuantizedWeights` for every attention
projection (`wq`, `wk`, `wv`, and `wo`) and MLP projection (`w_gate`, `w_up`,
and `w_down`). Replace `linear(x, w)` with `quantized_linear(x, w)`. In the Week
2 model loader, use `QuantizedWeights.from_mlx_layer(...)` instead of
materializing a 16-bit matrix. Keep the Week 1 model's readable boundary; its
layers still expect plain `mx.array` weights.

For embeddings, wire the `QuantizedEmbedding` from Task 1 into the loader:
load `embed_tokens` with `QuantizedWeights.from_mlx_layer(...)` and pass it to
`QuantizedEmbedding`. If the model has a separate `lm_head`, keep that head as
`QuantizedWeights` too and apply it with `quantized_linear`; `lm_head` is a
projection, not an embedding lookup.

Normalize each loaded layer's scales and biases to BF16. Require scales,
biases, and activations to match and return BF16. If the output is `nan` or
otherwise invalid, check for a dtype mismatch first.

Preserve the quantized layer's parameters as well. The model should pass
`w.group_size` and `w.bits` to the extension, which should validate the course
assumptions: `group_size = 128` and `bits = 4`.

You can test your implementation by running:

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-4b
```

You can also benchmark throughput and compare your implementation with the reference solution:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

Compare this result with the Day 1 `kv-cache` row. Do not start the decode
attention chapter until the complete model uses packed weights and the
end-to-end number has been recorded. The vanilla matrix product remains
callable as a correctness oracle, but only the SIMD matvec is integrated into
decode. Reprofile the quantized checkpoint across increasing cached context.
Day 4 is justified when attention's share grows with context after projection
traffic has fallen; otherwise keep tuning the measured matvec bottleneck.

{{#include copyright.md}}
