# Week 2 Day 3: Quantized Matvec

In this chapter, we will study and implement quantized matrix multiplication. Quantizing
weights from 16-bit floating point to 4-bit integers reduces both model size and
the memory traffic required for each generated token.

**📚 Readings**

- [Model Compression and Quantization](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [MLX Extensions Development Guide](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [Quantized Matmul on CPU (Video)](https://www.youtube.com/watch?v=es6s6T1bTtI)
- [Quantized Matmul on GPU (Video)](https://www.youtube.com/watch?v=jYCxVirq4d0)

## Why Quantization?

The decode phase of LLM inference is typically **memory-bandwidth bound**: each
token requires reading the model's weights but performs relatively little work
with them. The following rough calculation illustrates this for Qwen3-0.6B:

```plain
Per-token linear layers in decode phase:
- Input: 1 token × 1024 dimensions = 1024 bfloat16 values = 2 KB
- MLP weights: 1024 × 3072 × 3 matrices × 2 bytes = ~19 MB per layer
- Attention weights:
  - q_proj / o_proj: 1024 × 2048 × 2 matrices × 2 bytes = ~8 MB per layer
  - k_proj / v_proj: 1024 × 1024 × 2 matrices × 2 bytes = ~4 MB per layer
- Total weights per layer: ~31 MB
- Total for 28 layers: ~880 MB

FLOPs (2 per multiply-accumulate):
- MLP per layer: 2 × 3 × 1024 × 3072 ≈ 19M
- Attention projections per layer: 2 × (1024 × 2048 × 2 + 1024 × 1024 × 2) ≈ 13M
- 28 layers: ~880 million per token

Memory access: ~880 MB
Arithmetic intensity: 880M FLOPs / 880 MB ≈ 1.0 FLOPs/Byte
```

With M3 Max's 400 GB/s memory bandwidth and ~10 TFLOPS compute:

```plain
Memory-bound throughput: 400 GB/s × 1.0 FLOPs/Byte = 400 GFLOPS
Compute-bound throughput: 10 TFLOPS
```

This workload can use only about 4% of the available compute before exhausting
memory bandwidth.

### The Solution: Quantization

Compressing weights from 16-bit floating point (`float16` or `bfloat16`) to
4-bit integers (int4) can:

- **Reduce memory traffic by 4×**: 880 MB → ~220 MB per token
- **Improve arithmetic intensity by 4×**: 1.0 → ~4.0 FLOPs/Byte
- **Increase throughput by ~4×**: 400 GFLOPS → ~1.6 TFLOPS

The tradeoff is a small loss in model accuracy when the weights are quantized
carefully.

### Group-wise Quantization

Instead of applying one scale to an entire weight matrix, we divide each row
into **groups** and quantize every group independently. Local scales and biases
preserve more information about each group's weight distribution.

For a weight matrix $W$ of shape $(K, N)$, divide each row into groups of size
$G$. The Qwen3 MLX 4-bit checkpoints used in this course have a fixed group size
of 128:

```plain
Original weight matrix W: K × N (float16 or bfloat16)

Group size: G = 128
Number of groups per row = N / G

For each group of G consecutive values in a row:
  1. Find min and max values
  2. Compute scale and bias to map [min, max] → [0, 15] (4-bit range)
  3. Quantize each value using: quantized = round((value - bias) / scale)
```

All quantized-matmul tests use `group_size = 128`. They cover both `float16` and
`bfloat16` because MLX checkpoints may store their scales, biases, and
activations in either 16-bit data type.

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
Original: K × N float16/bfloat16 (2 bytes each) = 2KN bytes
Quantized: K × N int4 (0.5 bytes each) = 0.5KN bytes

Packing: 8 × 4-bit values fit in one uint32 (32 bits)

Weight matrix shape: K × N
Quantized storage shape: K × (N / 8) uint32
Scales shape: K × (N / G) float16/bfloat16
Biases shape: K × (N / G) float16/bfloat16
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

- $A$: shape $(M, N)$, float16 or bfloat16 (activations)
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
  A: M × N (float16 or bfloat16, activations)
  B_quantized: K × (N/8) (uint32, packed weights)
  scales: K × (N/G) (float16/bfloat16)
  biases: K × (N/G) (float16/bfloat16)

Output:
  C: M × K (float16/bfloat16)

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
  
  C[i, k] = float16/bfloat16(sum)
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
| `scales` | $(K, N/G)$ float16/bfloat16 | Per-group scale factors for dequantization. Each group of $G$ consecutive values shares one scale. Recall: $\text{scale} = (v_{max} - v_{min}) / 15$ |
| `biases` | $(K, N/G)$ float16/bfloat16 | Per-group bias (offset) for dequantization. Recall: $\text{bias} = v_{min}$ |
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

## Task 2: Implement a CPU Primitive

Implement quantized matrix multiplication as an MLX C++ extension. Follow the
existing `axpby` example: read `axpby.h`, `axpby.cpp`, and its binding in
`bindings.cpp` before starting.

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
  inputs, determine the output shape, and return a lazy `mx::array`. Implement
  `eval_cpu` to allocate the output, register arrays with the CPU encoder, and
  dispatch the compute kernel.
- **`CMakeLists.txt`** — Add the new C++ source to the extension target.

Follow `axpby`'s CPU encoder pattern in `eval_cpu`: allocate memory with
`out.set_data(mx::allocator::malloc(out.nbytes()))`, register the input and
output arrays, then dispatch a lambda for the computation. Inside the lambda,
implement the nested loop from the computation flow above. For each output
element `(i, k)`, unpack and dequantize the weights, accumulate products in
`float`, and write a result whose dtype matches the activation input.

Write the CPU implementation as a template, following `axpby`'s dtype-dispatch
pattern. Dispatch with `mx::float16_t` or `mx::bfloat16_t` according to the
output dtype.

The extension API is infrastructure: it lets an `mx.array` graph node schedule
the C++ loop you wrote. MLX owns the array lifetime and command encoder, but it
does not supply the quantized multiplication.

Build and test the extension:

```bash
pdm run build-ext
pdm run test --week 2 --day 3 -- -k task_2
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
   the direct GPU translation of the CPU loop and the correctness baseline.
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

### Prefill Tiling Moves to Week 3

Prefill has many activation rows and benefits from a different matrix-matrix
schedule. Week 2 keeps the vanilla kernel for that shape so its required path
stays focused on decode. The optional Week 3 performance lab adds 8×8
`simdgroup_matrix` tiles and studies reuse, barriers, register pressure, and
threadgroup scheduling.

### Stage 2: SIMD Matvec

Decode normally has `M = 1`; an 8×8 matrix tile would leave most rows empty.
Instead, one SIMD group reduces the input dimension and uses `simd_sum` to
combine lane-local partial sums. The regular projection kernel calculates two
output columns per SIMD group. The much wider tied vocabulary projection uses
eight columns so each activation load is reused across more weights.

The wide path also uses the affine identity

$$
\sum_j a_j(sq_j+b) = s\sum_j a_jq_j + b\sum_j a_j
$$

to avoid applying the bias separately to every unpacked value. Do not apply a
micro-optimization blindly to every shape: applying this rearrangement to the
two-output projection kernel reduced the measured full-model result from about
249 to 244.5 tok/s, while the extra live accumulators that help the vocabulary
projection also make smaller projections more difficult to schedule.

The scheduling numbers in this section are historical one-factor ablations run
against a completed model while developing the kernel. They explain why the
retained schedule looks this way; they are not Day 3's cumulative course
result. The end of this chapter reports the matched Day 2 to Day 3 checkpoint
change.

Likewise, raising the ordinary projection tile from two to four output columns
increased register pressure enough to reduce decode to about 232 tok/s. The
best measured split is two columns for normal projections and eight only for
the unusually wide vocabulary projection.

The first matvec copied the 2 KB activation vector into threadgroup memory and
waited at a barrier. Removing that copy was faster: the vector is cache-hot,
while every projection paid the synchronization cost. The measured decode
rate rose from roughly 238.6 to 246-249 tok/s. This is GPU scheduling in
practice: dispatch shape, occupancy, cache behavior, and barriers are part of
the algorithm.

After removing the barrier, grouping four SIMD groups per threadgroup instead
of eight was effectively neutral to slightly slower (about 248.5 tok/s in two
runs). Independent work does not guarantee a win from smaller threadgroups;
the extra threadgroups also add scheduling overhead. Keep eight for this M1
Pro workload and remeasure on different hardware.

Doubling only the wide vocabulary kernel to sixteen SIMD groups also regressed
the final run from 250.6 to 247.2 tok/s. Fewer threadgroups did not compensate
for the larger 512-thread scheduling unit, so the retained kernel uses eight
groups for both projection shapes.

### Direct Quantized Embedding Moves to Week 3

Week 2 performs row lookup and dequantization with basic `mlx.core` array
operations. The optional Week 3 performance lab fuses row gather, int4
unpacking, and affine dequantization into one Metal dispatch.

### Kernel Requirements

Implement both required kernel layouts in `quantized_matmul.metal`:

- First, implement the vanilla one-thread-per-output matrix grid.
- For `M <= 8`, assign one SIMD group to an output tile. Cooperatively reduce
  the input dimension and compute several output columns per group.
- The kernel should support both `half` and `bfloat16_t` inputs and outputs.
- Apply the same group-wise dequantization loop as the CPU version:
  - Iterate over groups of 128 values.
  - Unpack int4 values from each `uint32`.
  - Dequantize each value with `q * scale + bias`.
  - Accumulate products in `float`, then cast the result to the kernel dtype.
- Add boundary checks (`i < M`, `k < K`) before writing output.

The custom kernel only needs to support `bits = 4` and `group_size = 128`. Use
the group size to compute `groups_per_row` and the packed-weight offsets.
Instantiate the templated Metal kernel once for `half` and once for
`bfloat16_t`; select the matching kernel name in `eval_gpu`.

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
   so packed input values and activations can be reused.
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

Qwen3 MLX quantized layers may use either **float16** or **bfloat16** for the
tensors involved in dequantization. Accept either dtype, require `scales`,
`biases`, and activations to match, and return that same dtype. If the output is
`nan` or otherwise invalid, check for a dtype mismatch first.

Preserve the quantized layer's parameters as well. The model should pass
`w.group_size` and `w.bits` to the extension, which should validate the course
assumptions: `group_size = 128` and `bits = 4`.

You can test your implementation by running:

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-0.6b
```

You can also benchmark throughput and compare your implementation with the reference solution:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

Compare this result with the Day 1 `kv-cache` row. Do not start the decode
attention chapter until the complete model uses packed weights and the
end-to-end number has been recorded. The vanilla matrix product remains
callable as a correctness oracle, but only the SIMD matvec is integrated into
decode.

## Expected Performance Contribution

**Measured decode-kernel improvement: about 1.4-1.9x for ordinary projections
and about 10.5x for the vocabulary head over the scalar quantized kernel.** In
a Qwen3-0.6B M4 Pro run, the Q/K/V/O projection speedups ranged from 1.39-1.65x,
the gate/up/down projections from 1.86-1.89x, and the 151,936-row tied output
head reached about 10.5x. In the cumulative M4 Pro course ladder, integrating
packed weights and the retained SIMD matvec increased decode from 101.80 to
134.24 tok/s (+31.9%): 6.88x Week 1 and 59.1% below MLX. On the earlier M1 Pro
reference, the first two-column SIMD matvec improved the then-current path by
about 5.7%; removing its activation barrier later added roughly another 3-4%.

Prefill tiling and direct quantized embedding are measured in the optional
Week 3 performance lab. They are intentionally excluded from the minimal Week
2 acceptance path. Percentages from later chapters overlap with this one and
must not be added together.

{{#include copyright.md}}
