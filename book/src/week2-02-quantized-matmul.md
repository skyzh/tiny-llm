# Week 2 Day 2-3: Quantized Matmul

In this chapter, we will implement the quantized matrix multiplication. Quantization compresses model weights from 16-bit floating point to 4-bit integers, which is critical for efficient LLM serving on devices with limited memory bandwidth.

**📚 Readings**

- [Model Compression and Quantization](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [MLX Extensions Development Guide](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [Quantized Matmul on CPU (Video)](https://www.youtube.com/watch?v=es6s6T1bTtI)
- [Quantized Matmul on GPU (Video)](https://www.youtube.com/watch?v=jYCxVirq4d0)

## Why Quantization?

As we learned in the KV Cache chapter, the decode phase of LLM inference is **memory-bandwidth bound**. Let's revisit the arithmetic intensity calculation for the Qwen2-0.5B model:

```plain
Per-token computation in decode phase:
- Input: 1 token × 896 dimensions = 896 float16 values = 1.792 KB
- MLP weights: 896 × 4864 × 3 matrices × 2 bytes = ~25 MB per layer
- Attention weights: 896 × 896 × 4 matrices × 2 bytes = ~6 MB per layer
- Total weights per layer: ~31 MB
- Total for 24 layers: ~750 MB

FLOPs (2 per multiply-accumulate):
- MLP per layer: 2 × 3 × 896 × 4864 ≈ 26M
- Attention per layer: 2 × 4 × 896 × 896 ≈ 6.4M
- 24 layers: ~780 million per token

Memory access: ~750 MB
Arithmetic intensity: 780M FLOPs / 750 MB ≈ 1.0 FLOPs/Byte
```

With M3 Max's 400 GB/s memory bandwidth and ~10 TFLOPS compute:

```plain
Memory-bound throughput: 400 GB/s × 1.0 FLOPs/Byte = 400 GFLOPS
Compute-bound throughput: 10 TFLOPS

We're using only ~4% of available compute!
```

### The Solution: Quantization

By compressing weights from 16 bits (float16/bfloat16) to 4 bits (int4), we:

- **Reduce memory bandwidth by 4×**: 750 MB → ~190 MB per token
- **Improve arithmetic intensity by 4×**: 1.0 → ~4.0 FLOPs/Byte
- **Increase throughput by ~4×**: 400 GFLOPS → ~1.6 TFLOPS

The tradeoff is minimal accuracy loss with proper quantization techniques.

### Group-wise Quantization

Instead of quantizing all weights uniformly, we divide them into **groups** and quantize each group independently. This preserves more information about the weight distribution.

For a weight matrix $W$ of shape $(K, N)$, we divide each row into groups of size $G$ (typically 64 or 128):

```plain
Original weight matrix W: K × N (float16/bfloat16)

Group size G = 64
Number of groups per row = N / G

For each group of 64 consecutive values in a row:
  1. Find min and max values
  2. Compute scale and bias to map [min, max] → [0, 15] (4-bit range)
  3. Quantize each value using: quantized = round((value - bias) / scale)
```

### Affine Quantization

We use **affine (asymmetric) quantization** which maps a floating-point range to the full integer range:

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

For efficient storage and computation, quantized weights are packed:

```plain
Original: K × N float16 (2 bytes each) = 2KN bytes
Quantized: K × N int4 (0.5 bytes each) = 0.5KN bytes

Packing: 8 × 4-bit values fit in one uint32 (32 bits)

Weight matrix shape: K × N
Quantized storage shape: K × (N / 8) uint32
Scales shape: K × (N / 64) float16
Biases shape: K × (N / 64) float16
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

- $A$: shape $(M, N)$, float16/bfloat16 (activations)
- $B$: shape $(K, N)$, **quantized** to int4 (weights)
- $C$: shape $(M, K)$, float16/bfloat16 (output)

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

This shows we can factor out the scale and bias per group, reducing the number of floating-point operations.

### Computation Flow

```plain
Input:
  A: M × N (float16, activations)
  B_quantized: K × (N/8) (uint32, packed weights)
  scales: K × (N/64) (float16)
  biases: K × (N/64) (float16)

Output:
  C: M × K (float16)

For each output element C[i, k]:
  sum = 0
  for each group g in 0..(N/64 - 1):
    scale = scales[k, g]
    bias = biases[k, g]
    
    # Process 64 values in the group (8 uint32 packs)
    for each pack p in 0..7:
      packed_value = B_quantized[k, g*8 + p]
      
      # Unpack 8 × 4-bit values
      for bit_offset in [0, 4, 8, 12, 16, 20, 24, 28]:
        quantized = (packed_value >> bit_offset) & 0xF
        b_value = quantized * scale + bias
        a_value = A[i, g*64 + p*8 + bit_offset/4]
        sum += a_value * b_value
  
  C[i, k] = sum
```

## Task 1: Implement QuantizedWeights

```
src/tiny_llm/quantize.py
```

First, familiarize yourself with the `QuantizedWeights` class, which stores quantized weight information:

| Field | Shape | Description |
|-------|-------|-------------|
| `weight` | $(K, N/8)$ uint32 | Packed quantized weights. Each uint32 stores 8 consecutive 4-bit values. The original weight matrix has shape $(K, N)$, and after packing, it becomes $(K, N/8)$. |
| `scales` | $(K, N/G)$ float16 | Per-group scale factors for dequantization. Each group of $G$ consecutive values shares one scale. Recall: $\text{scale} = (v_{max} - v_{min}) / 15$ |
| `biases` | $(K, N/G)$ float16 | Per-group bias (offset) for dequantization. Recall: $\text{bias} = v_{min}$ |
| `group_size` | int | Number of consecutive values that share the same scale/bias (typically 64) |
| `bits` | int | Quantization bit width (typically 4, meaning values are in range $[0, 15]$) |

The `from_mlx_layer` static method extracts these fields from MLX's quantized linear layers when loading the model.

Next, implement the `quantized_linear` function, which is a wrapper around `quantized_matmul` that mimics the standard `linear` function interface. And we'll implement `quantized_matmul` in the next task.

## Task 2: Implement `quantized_matmul` (CPU version)

In this task, we will implement the quantized matmul as an MLX C++ extension. The pattern is identical to the existing `axpby` example in the codebase — read through `axpby.h`, `axpby.cpp`, and the corresponding binding in `bindings.cpp` first as your reference.

```
src/extensions/src/tiny_llm_ext.h
src/extensions/bindings.cpp
src/extensions/src/quantized_matmul.cpp
src/extensions/CMakeLists.txt
```

You need to touch three files, all within the `tiny_llm_ext` namespace:

- **`tiny_llm_ext.h`** — Declare the `quantized_matmul(...)` function signature and define a `QuantizedMatmul` primitive class (inheriting `mx::Primitive`). Store `group_size` and `bits` as private members.
- **`bindings.cpp`** — Add an `m.def(...)` call to expose the function to Python.
- **`quantized_matmul.cpp`** — Implement the `quantized_matmul(...)` function (validate inputs, compute output shape, return a lazy `mx::array`) and the `eval_cpu` method (allocate output, register arrays with the CPU encoder, dispatch the compute kernel).

The `eval_cpu` implementation follows the same CPU encoder pattern as `axpby`: allocate output memory with `out.set_data(mx::allocator::malloc(out.nbytes()))`, register input/output arrays with the encoder, then dispatch a lambda that performs the actual computation. Inside the lambda, implement the nested loop from the Computation Flow section above — iterate over each output element `(i, k)`, accumulate in `float` (fp32) to avoid precision loss, and cast the result back to `float16` when writing to the output.

Don't forget to add `src/quantized_matmul.cpp` to `target_sources` in `CMakeLists.txt`.

You can test your implementation by running:

```bash
pdm run build-ext
pdm run test --week 2 --day 2 -- -k task_2
```

## Task 3: Implement `quantized_matmul` (GPU version)

```
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

In this task, you will write the Metal kernel for quantized matmul **and** wire up the `eval_gpu` method to dispatch it. Keep the math exactly the same as Task 2 (CPU); only the execution model changes.

### Metal Kernel

You need to implement one kernel entry in `quantized_matmul.metal`:

- Use a **one-thread-per-output-element** mapping: each thread computes `out[i, k]`.
- The kernel should be templated on the data type (to support both `half` and `bfloat16_t`).
- Apply the same group-wise dequantization loop as the CPU version:
  - Iterate over groups (`group_size = 64`)
  - Unpack int4 values from packed `uint32`
  - Dequantize with `q * scale + bias`
  - Accumulate in `float` and cast to the output dtype at the end
- Add boundary checks (`i < M`, `k < K`) before writing output.

### GPU Dispatch

Complete the `eval_gpu` method in `quantized_matmul.cpp` to dispatch your Metal kernel. Follow the same pattern as `axpby`'s GPU dispatch:

1. Get the Metal device and command encoder from the stream.
2. Select the correct kernel name based on the activation dtype (`float16` → `half`, `bfloat16` → `bfloat16_t`).
3. Set input/output buffers and dimension constants (`M`, `N`, `K`) on the encoder — make sure the buffer order matches your kernel signature.
4. Calculate a 2D threadgroup configuration: use `kernel->maxTotalThreadsPerThreadgroup()` to determine the total threads, then split between the M and K dimensions (e.g., 32 threads for M, the rest for K).
5. Dispatch with `dispatchThreadgroups`.

You can test your implementation by running:

```bash
pdm run build-ext
pdm run test --week 2 --day 2 -- -k task_3
```

## Task 4: Model Integration

```
src/tiny_llm/qwen2_week2.py
```

Integrate your quantized matmul into the Week 2 Qwen2 model so that inference runs on quantized weights end-to-end.

Change the weight type from `mx.array` to `QuantizedWeights` for all linear layers in attention (`wq/wk/wv/wo`) and MLP (`w_gate/w_up/w_down`). Replace every `linear(x, w)` call with `quantized_linear(x, w)`. In the model loading code, use `QuantizedWeights.from_mlx_layer(...)` to extract quantized weight information from each MLX linear layer, instead of calling `mx.dequantize` to get a full float16 matrix. Make sure the Week 1 loader still dequantizes (since Week 1 layers expect plain `mx.array`), while the Week 2 loader does **not** dequantize.

Note that MLX loads quantized models with `scales` and `biases` stored in **bfloat16** by default, while the activation tensors are typically **float16**. Since we have not implemented bfloat16 support in our kernel, you will need to convert the scales and biases to float16 with `mx.astype` before calling the kernel. If you see `nan` or garbage output, a dtype mismatch is the most likely cause.

You can test your implementation by running:

```bash
pdm run main --solution tiny_llm --loader week2 --model qwen2-0.5b
```

{{#include copyright.md}}
