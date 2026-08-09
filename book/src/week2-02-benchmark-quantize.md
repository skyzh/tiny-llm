# 🚧 Week 2 Day 2: Benchmark, Profile, and Quantize

> **Status: Experimental.** See the
> [Week 2 verification matrix](./week2-overview.md#verification-status) for
> what is continuously tested, locally measured, and still under review.

Day 1 gave us a cached model. Day 2 starts by measuring it: how fast is decode,
and which operator family owns the time? The profile points at projection weight
reads as the largest removable cost. The rest of the chapter replaces dense
16-bit weights with a 4-bit quantized representation and implements the
matrix-vector kernel that operates on it directly.

**📚 Readings**

- [Model Compression and Quantization](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [MLX Extensions Development Guide](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [Quantized Matmul on GPU (Video)](https://www.youtube.com/watch?v=jYCxVirq4d0)

## Benchmark the Cached Model

Optimization starts with a trustworthy comparison. Prefill processes many prompt
tokens at once; decode usually processes one token per request and is dominated
by repeatedly reading quantized weights. A change can improve one phase while
hurting the other, so `benches/bench.py` reports both:

- prefill tokens per second: prompt tokens divided by prefill time;
- decode tokens per second: generated tokens after the first token divided by
  decode time.

The first generated token belongs to prefill. Excluding it from decode prevents
prompt length from distorting the decode number.

Choose the prefill workload before comparing implementations. Prompt scoring
needs logits for every position, while serving needs only the final prompt
logit. Use `--prefill-logits all` for the former and
`--prefill-logits last` for the latter. The runner applies the choice to your
solution and MLX alike. Never compare a final-row run from your solution with an
all-row MLX run.

Both sides of the Week 2 comparison use a KV cache: prefill the prompt once,
then pass only the newly generated token on each decode step. Comparing a cached
MLX baseline with your solution recomputing the full prefix would measure two
different algorithms and make the kernel target meaningless.

### Record a Matched Baseline

Use the same model, prompt length, output length, device, and warmup count for
your solution and MLX:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2 \
  --prefill-logits last

pdm run bench --solution mlx --loader week2 --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2 \
  --prefill-logits last
```

Use `--solution tiny_llm_ref` with the same arguments when you want to compare
your solution with the reference solution instead of MLX.

Or run the cumulative ladder in fresh processes:

```bash
pdm run bench-week2-progression --offline --repeats 4 \
  --solution tiny_llm \
  --variant week2-kv-cache --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-baseline.json
```

Benchmark on an otherwise idle machine: stop other CPU- and GPU-intensive
workloads, keep power mode and ambient conditions fixed, and let the machine
return to a stable temperature before comparing runs. Run each command several
times, report the median, and include the hardware, MLX and mlx-lm versions,
prefill-logit mode, and exact model with the result. A dependency upgrade
changes the comparison baseline, so remeasure MLX rather than carrying an old
denominator forward.

### Synchronize Lazy Work

MLX builds lazy computation graphs. Timing only the Python call measures graph
construction, not GPU execution. Every timed iteration must evaluate the output:

```python
start = perf_counter()
output = function()
mx.eval(output)
elapsed = perf_counter() - start
```

The benchmark must also call the cache release hook after warmups and timed
runs so cache implementations with owned or shared resources can return them:

```bash
pdm run test --week 2 --day 2
```

### Debug Metal Without a CPU Twin

> **Note:** Day 3 onward adds GPU-only extensions. A C++ CPU version is
> possible but not required. Use the three-level validation ladder below
> instead.

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
expected = readable_operator(*inputs)
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

Metal API Validation and an Xcode GPU capture can help diagnose dispatch and
resource problems, but they supplement this ladder rather than replace its
small deterministic comparisons. Only profile after the vanilla and optimized
kernels agree with the Python reference oracle.

### Benchmark Protocol (Reference)

The isolated operator benchmarks use the same synchronization rule. Evaluate
input setup before invoking the benchmark fixture so setup does not leak into
the result. The Week 2 operator ladder compares the Python reference equation,
the optimized kernel in your solution, and MLX at the selected model's real
tensor shapes:

```bash
pdm run bench-week2-operators --solution tiny_llm \
  --model qwen3-4b --context 128 --section decode-projections
```

The runner rotates through every implementation order so thermal or GPU
performance-state drift does not consistently favor one path. Choose enough
warmup iterations to exclude compilation, synchronize every timed iteration,
and repeat the run in fresh processes. Report the median with the exact
hardware, dependency versions, model, and tensor shapes. The
[performance appendix](./appendix-performance.md) applies this protocol to the
reference-solution checkpoints and keeps the resulting machine-specific numbers
in one place.

To rank complete model work without requiring a GUI, replay your current
kernel groups at Qwen3-4B shapes and dispatch counts:

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --warmup 4 --iterations 12 \
  --json-output week2-kernel-profile.json
```

The projection group preserves the transformer dependency order, including the
attention projections before the output projection and the MLP after the
attention residual. This matters for occupancy: making every layer independent
would let unrelated work hide an under-filled kernel and produce a false
Split-K conclusion. Each measured round rotates the group order, synchronizes
once per group, and normalizes the group medians into an attribution profile.

The resulting shares are not a throughput benchmark. Group boundaries force
materialization that a complete lazy graph may fuse, while a capture adds its
own overhead. Use the profile to rank kernel groups, then require the ordinary
fresh-process model benchmark to confirm the change.

### Optional Advanced Investigation

The required Day 2 lab ends after you save the fresh-process benchmark JSON and
the dependency-aware operator attribution. If those two measurements disagree,
or if you want to investigate a course-owned Metal shader, continue with the
[advanced Metal profiling appendix](./week2-advanced-profiling.md).

That appendix contains the `.gputrace` capture and Xcode GUI replay workflow,
the consistent screenshot checklist for Pipeline Statistics, memory, and
Shader Cost Graph evidence, Instruments commands, and the evidence order for
schedule tuning. Keeping those steps out of the required lab makes the
boundary explicit: they are useful performance research, not prerequisites
for understanding prefill, decode, synchronization, or matched baselines.

### Optional Performance-Lab Acceptance Target

The performance-lab targets are:

```plain
your solution's prefill throughput / MLX prefill throughput >= 0.80
your solution's decode throughput / MLX decode throughput >= 0.80
```

Both ratios use Qwen3-4B, a 128-token prompt,
128 timed decode steps, and last-row logits. `--output-len 129` includes the
first token produced by prefill. Reaching 80% is the optional performance-lab
threshold, not a promise that every educational kernel individually matches
its MLX counterpart. MLX is the comparison baseline; the performance-lab
solution must reach both targets with its own operator implementations. If either
ratio misses, the next chapter starts from the new benchmark and profile rather
than a predetermined optimization.

Keep a 2K context run in the report as a stress diagnostic. It is useful for
showing when attention overtakes fixed-shape projections, but changing context
also changes the problem. Do not move the acceptance shape after seeing a
result.

## Why Quantize: The Decode Roofline

Profile only the checkpoint you have completed. The decode phase of LLM
inference is typically **memory-bandwidth bound**: each token requires reading
the model's weights but performs relatively little work with them. Attach two
results to the checkpoint report: the fresh-process JSON with your solution and
MLX, and the kernel-group JSON with absolute times as well as shares. The first
says how far decode is from MLX; the second says which operator family owns the
current implementation's time.

Use the dimensions in the official
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
layers, not a prediction of complete-model throughput. A dense FP16 or BF16
weight occupies two bytes:

```plain
4,022,272,000 weights × 2 bytes = 8.045 GB per token
arithmetic intensity = 8.045 GFLOPs / 8.045 GB = 1.0 FLOP/byte
```

FP16 and BF16 divide their 16 bits differently: FP16 gives more bits to the
significand, while BF16 gives more bits to the exponent. That affects numerical
range and precision, but not this bandwidth calculation. The course uses BF16
for activations and outputs.

| Dense weight format | Bits per weight | Bytes per weight | Streamed weight bytes per token | Weight arithmetic intensity |
|---|---:|---:|---:|---:|
| FP16 | 16 | 2 | 8.045 GB | 1.0 FLOP/byte |
| BF16 | 16 | 2 | 8.045 GB | 1.0 FLOP/byte |

This is the baseline to improve: both dense formats must stream roughly 8 GB of
projection weights to generate one token.

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

Day 5 briefly revisits row lookup as an optional one-dispatch fusion; keep the
Day 2 implementation simple and explicit.

## Task 2: Migrate the Cached Model to Quantized Weights

```
src/extensions/src/tiny_llm_ext.h
src/extensions/bindings.cpp
src/extensions/src/quantized_matmul.cpp
src/extensions/CMakeLists.txt
```

Register quantized matrix multiplication as an MLX C++ extension. Follow the
existing `axpby` example for array validation, lazy primitive construction,
bindings, and Metal dispatch. Your solution is GPU-only; its `eval_cpu` method
should raise a clear unsupported-device error.

You will update four files. Keep the C++ declarations and definitions in the
`tiny_llm_ext` namespace:

- **`tiny_llm_ext.h`** — Declare the `quantized_matmul(...)` function signature
  and define a `QuantizedMatmul` primitive class (inheriting `mx::Primitive`).
  Store `group_size` and `bits` as private members.
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
pdm run test --week 2 --day 2 -- -k task_1
```

## Task 3: Implement Metal Matrix Products

Before writing your first Metal kernel, understand the execution model. Metal
organizes GPU work in three nested scopes:

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
  of threadgroups; the GPU schedules them across available cores. More
  threadgroups expose more parallelism, but each threadgroup consumes
  registers and threadgroup memory. Too many threadgroups per core reduces
  occupancy — fewer in-flight threadgroups mean less ability to hide memory
  latency by switching between them.

Key tradeoff: when you increase the work per SIMD group (more output columns,
larger tiles), you increase register pressure and can reduce occupancy. Start
with one SIMD group per threadgroup and profile. Add threadgroups only when
the measured gain justifies the resource cost. "Start with one SIMD group, then
profile" is a launch-shape strategy, not just a debugging suggestion.

```
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

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
dimension. Day 2 uses this explicit dispatch:

| Activation rows | Kernel | Role at this checkpoint |
|---:|---|---|
| `M <= 8` | SIMD matvec | Optimized path for decode and other very small matrix inputs. |
| `M > 8` | Vanilla matmul | Correctness-first prefill path; Day 5 replaces it with a cooperative tiled kernel. |

The cutoff does not mean the SIMD kernel expands to cover larger `M`. The two
paths are separate schedules: Day 2 optimizes the vector-shaped decode
bottleneck and leaves matrix-shaped prefill visible for the later profile to
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

Keep the vanilla kernel for matrix-shaped prefill in this chapter; Day 5
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
  SIMD matvec schedule; Day 5 introduces the tiled prefill schedule.
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
pdm run test --week 2 --day 2 -- -k gpu
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

## Benchmark Analysis: Select Day 3

Measure the cumulative model, the real projection shapes, and the updated
operator attribution:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-kv-cache --variant week2-quantized-matvec --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last

pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section decode-projections --context 128

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case quantized-matvec:decode:128 --warmup 4 --iterations 12
```

Use this optional advanced capture only when you are tuning the shader or the
model and operator results disagree:

```bash
CMAKE_ARGS="-DMLX_METAL_DEBUG=ON" pdm run build-ext
MLX_METAL_DEBUG=1 MTL_CAPTURE_ENABLED=1 pdm run capture-week2-shader \
  --solution tiny_llm --workload quantized-projection \
  --projection q --rows 1 --schedule matvec --iterations 10 \
  --output /tmp/week2-day2-packed-q-m1.gputrace
```

Attach the complete-model before/after rows, the per-projection latency table,
and the new kernel-group profile. First require a clear decode gain over
`kv-cache`. Then compare each projection with MLX at the identical shape.
Projections may remain the largest absolute category because the model performs
them in every layer; once their operator latency is close to MLX, that bar is
no longer the largest removable gap.

The [Xcode checkpoint contract](./appendix-performance.md#week-2-xcode-checkpoint-contract)
describes how to inspect the pipeline, limiters, memory traffic, and
highest-cost source lines if you generate the optional trace.
Continue to Day 3 when the matched projection table is close to MLX and the
post-Day-2 profile makes normalization, position, and activation the largest
removable gap. If the projection comparison is still far behind, keep tuning
the matvec instead. The
[reference checkpoint](./appendix-performance.md#day-2-quantize-weights)
pairs the model delta, projection microbenchmarks, and attribution. Its matched
operator result is the reason the hot matvec products do not remain the next
target: after the projection gap shrinks, the pointwise cluster is the larger
removable model cost.

{{#include copyright.md}}
