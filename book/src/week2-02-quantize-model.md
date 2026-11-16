# 🚧 Week 2 Day 2: Packed W4 Weights

At the end of [Day 1](./week2-01-kv-cache.md) your model reuses its K/V cache,
so each decode step feeds only one new token through the network. That token
is cheap to compute, but it still has to visit every projection weight in the
model. Day 1's roofline showed what that costs: for Qwen3-4B, one decode step
streams about 8 GB of BF16 weights and does only one floating-point operation
per byte it reads. Decode spends its time waiting on memory.

That is today's hypothesis, and you can check it on your own model first.
After Day 1 you saved a request benchmark and a decode replay of
`kv-cache` (see [Day 1](./week2-01-kv-cache.md#find-where-the-time-goes)).
If decode takes most of the request and the projections group leads the
decode replay, the hypothesis fits your machine: each decode step's time is
set by the weight bytes it reads, so reading fewer bytes should shrink the
projections group and speed up decode. If another group leads, write down
which one. You still need today's packed weights for Days 3 to 6, and the
comparison at the end of the day will show what packing actually did.

The checkpoint you download is already much smaller than that. The
`Qwen3-4B-MLX-4bit` files store each weight as a 4-bit code. Up to now the
course has unpacked those codes into BF16 when loading the model, then read
the unpacked copy on every token. Today you stop doing that. The weights stay
packed in memory, and a Metal kernel unpacks each one in registers right before
multiplying it by the activation. Every token then reads about a quarter of
the bytes.

By the end of the day, you will have:

1. an embedding lookup that unpacks only the rows it needs;
2. a quantized-matmul operator, backed by your own MLX primitive;
3. a simple Metal kernel for it, with one thread per output, that is easy to
   check against the math; and
4. a model checkpoint, `packed-weights`, whose projections and output head all
   run through that kernel, for prompts and decode alike.

The simple kernel is not fast. Today's question is only what keeping the
weights packed does on its own. [Day 3](./week2-03-decode-matvec.md) replaces
the kernel on the decode path with one built for a single activation row.

The starter already supplies the packed-weight container
(`QuantizedWeights` and its `from_mlx_layer` loader), the C++ declaration and
Python binding for the operator, CMake registration, and stubs that raise an
error naming the missing function. Build both extensions, then run the first
tests. They fail until you fill in the stubs:

```bash
pdm run build-ext
pdm run build-ext-ref
pdm run test --week 2 --day 2 -- -k task_1
pdm run test --week 2 --day 2 -- -k task_3
```

**📚 Readings**

- [Model Compression and Quantization](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [MLX Extensions Development Guide](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [Quantized Matmul on GPU (Video)](https://www.youtube.com/watch?v=jYCxVirq4d0)

## Eight Weights, Four Bits Each

Start with a small example. Suppose a row of a weight matrix begins with these
eight values:

```plain
w = [-0.50, -0.31, 0.02, 0.10, 0.33, 0.47, 0.61, 0.80]
```

Four bits give you 16 codes, `0` through `15`. One simple way to map the
values onto those codes is to spread the codes evenly between the smallest
and largest value:

```plain
bias  b = min(w)                   = -0.50
scale s = (max(w) - min(w)) / 15   =  0.08667
code  q = round((w - b) / s)

q = [0, 2, 6, 7, 10, 11, 13, 15]
```

To get a weight back, reverse the mapping:

$$
\hat{w} = q s + b
$$

```plain
q * s + b = [-0.500, -0.327, 0.020, 0.107, 0.367, 0.453, 0.627, 0.800]
```

Each reconstructed value is within half a step (`s / 2`, about 0.043) of the
original. That small error is the price of storing four bits per weight
instead of sixteen.

You will never run this quantization step yourself. The checkpoint already
contains the codes, scales, and biases, and the model may have chosen them
differently from the min/max rule above. Your code only does the
reconstruction, `q * s + b`, using the stored numbers exactly as they are.

### Groups of 128

A single scale and bias for a whole matrix row would be a poor fit: one large
weight would stretch the range and waste most of the 16 codes on values that
never occur. Instead, each row is split into **groups** of consecutive
weights, and every group gets its own scale and bias. The Qwen3 MLX 4-bit
checkpoints use a group size of 128, and every required test in this chapter
does too.

The format has a name: **W4A16**. Weights use 4-bit codes; activations stay
16-bit floating point. In this course, activations, scales, biases, and
outputs are all BF16.

### The Scale Can Be Negative

The codes are unsigned, but the stored scale is signed. With a positive scale,
code 0 is the low end of the group's range. With a negative scale, code 0 is
the high end:

```plain
positive scale: s =  0.0867, b = -0.5  =>  q=0 is -0.5, q=15 is about  0.8
negative scale: s = -0.0867, b =  0.8  =>  q=0 is  0.8, q=15 is about -0.5
```

Because a stored scale can be negative, you cannot assume code 0 is the low
end. If you rebuild the scale from an assumed min/max, every group with a
negative scale comes out reversed. Use `scale` and `bias` as stored.

### Eight Codes in One `uint32`

A 4-bit code is too small to address on its own, so eight codes share one
32-bit word. The first code sits in the lowest four bits:

```plain
codes  [a, b, c, d, e, f, g, h] = [0, 2, 6, 7, 10, 11, 13, 15]

word = (h << 28) | (g << 24) | (f << 20) | (e << 16)
     | (d << 12) | (c << 8)  | (b << 4)  | a
     = 0xFDBA7620

a = (word >> 0)  & 0xF = 0
b = (word >> 4)  & 0xF = 2
...
h = (word >> 28) & 0xF = 15
```

Read the hex digits from right to left and you get the codes in order:
`0, 2, 6, 7, A, B, D, F`.

For a matrix with `K` output rows and `N` input columns, the checkpoint stores
three arrays:

```plain
packed codes: K × (N / 8)    uint32    eight codes per word
scales:       K × (N / 128)  bfloat16  one per group
biases:       K × (N / 128)  bfloat16  one per group
```

The `QuantizedWeights` container in `src/tiny_llm/quantize.py` holds exactly
these fields:

| Field | Shape | Meaning |
|-------|-------|---------|
| `weight` | `(K, N/8)` uint32 | Packed codes, eight per word, lowest nibble first. |
| `scales` | `(K, N/G)` bfloat16 | Signed per-group scale. Its sign decides which end of the range code 0 maps to. |
| `biases` | `(K, N/G)` bfloat16 | Per-group offset. Code 0 reconstructs to this value. |
| `group_size` | int | `G`, the number of weights sharing one scale and bias: `128` here. |
| `bits` | int | Bits per code: `4` here, so codes run from 0 to 15. |

`QuantizedWeights.from_mlx_layer` reads these fields from an MLX quantized
layer and converts the scales and biases to BF16. Use it everywhere you load a
quantized layer rather than writing a second loader.

## How Much Faster Can Decode Get?

The packed codes are half a byte per weight, but each group of 128 also
carries a 2-byte scale and a 2-byte bias. Add that metadata and recompute
Day 1's numbers for Qwen3-4B:

```plain
bytes per W4 weight  = 0.5 + (2 + 2) / 128   = 0.53125 bytes
streamed bytes/token = 4,022,272,000 × 0.53125 = 2.137 GB
arithmetic intensity = 8.045 GFLOPs / 2.137 GB = 3.765 FLOPs/byte
```

| Weight format | Value bits | Metadata per 128 weights | Bytes per weight | Weight bytes per token | FLOPs per byte |
|---|---:|---|---:|---:|---:|
| FP16 | 16 | None | 2 | 8.045 GB | 1.0 |
| BF16 | 16 | None | 2 | 8.045 GB | 1.0 |
| W4 | 4 | One BF16 scale and one BF16 bias | 0.53125 | 2.137 GB | 3.765 |

Decode reads 3.765 times fewer projection-weight bytes per token. If decode
were limited only by reading those weights, that would also be the speedup.
Real decode also reads K/V, runs normalization and attention, and launches
many small operations, so expect a smaller gain.

### A Bandwidth Ceiling for Each Chip

Apple publishes memory bandwidth for each chip, but not a comparable BF16
GPU throughput figure. Bandwidth alone still gives a useful upper bound:

```plain
best possible tokens/s = memory bandwidth / weight bytes per token
```

The table below uses the highest-bandwidth configuration of each chip in
Apple's published specifications, with decimal GB as Apple uses. These
numbers are ceilings for the arithmetic above, not measurements.

| Chip | Bandwidth | FP16/BF16 ceiling | W4 ceiling |
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

Bandwidths come from Apple's specifications for
[M1 Pro and Max](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/),
[M1 Ultra](https://www.apple.com/newsroom/2022/03/apple-unveils-m1-ultra-the-worlds-most-powerful-chip-for-a-personal-computer/),
[M2 Pro and Max](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/),
[M2 Ultra](https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/),
[M3 Pro and Max](https://support.apple.com/en-us/117736),
[M3 Ultra in the 2025 Mac Studio](https://support.apple.com/en-us/122211), and
[M4 Pro and Max](https://support.apple.com/en-us/121553). The table stops at
M4 Pro and Max; it makes no claim about later chips.

A real model falls short of these ceilings because it reads more than
projection weights and cannot hold peak bandwidth for a whole step. The
[performance appendix](./appendix-performance.md) keeps measured results
separate from this exercise.

The roofline also applies only to decode, where one activation row meets each
weight. Prefill sends many rows through the same weights, so each byte is
reused many times and the bottleneck moves toward arithmetic. Day 4 gives
prefill its own kernel.

## The Math You Will Implement

The course writes a linear layer as `C = A Bᵀ`:

- `A`: activations, shape `(M, N)`, BF16;
- `B`: weights, shape `(K, N)`, stored as W4 codes;
- `C`: outputs, shape `(M, K)`, BF16.

`M` is the number of activation rows after you flatten every leading
dimension. During decode, `M` is 1. Each output is a dot product over the
input dimension:

$$
C[i, k] = \sum_{j=0}^{N-1} A[i, j] \cdot B[k, j]
$$

With quantized weights, `B[k, j]` is reconstructed from its code and its
group's parameters, where `g = floor(j / G)`:

$$
B[k, j] = q[k, j] \cdot \text{scale}[k, g] + \text{bias}[k, g]
$$

Split the sum by group. Inside a group the scale and bias are constants, so
they come out of the inner sum:

$$
C[i, k] = \sum_{g=0}^{N/G-1} \left( \text{scale}[k, g] \sum_{j'=0}^{G-1} A[i, gG + j'] \, q[k, gG + j'] \;+\; \text{bias}[k, g] \sum_{j'=0}^{G-1} A[i, gG + j'] \right)
$$

In words: for each group, multiply the scale by the dot product of activations
and raw codes, add the bias times the plain sum of those activations, and add
up the group results. Day 3's fast decode kernel uses exactly this form. For one group, the identity is:

$$
\sum_j a_j (s q_j + b) = s \sum_j a_j q_j + b \sum_j a_j
$$

Written as a loop, the direct version computes one output like this:

```plain
for each output C[i, k]:
  sum = 0.0                                  # float32 accumulator
  for each group g in 0 .. N/G - 1:
    scale = scales[k, g]
    bias  = biases[k, g]
    for each packed word p in 0 .. G/8 - 1:  # 16 words per group
      word = B_packed[k, g * (G/8) + p]
      for shift in [0, 4, 8, ..., 28]:
        q = (word >> shift) & 0xF
        a = A[i, g*G + p*8 + shift/4]
        sum += a * (q * scale + bias)
  C[i, k] = bfloat16(sum)
```

## Task 1: Unpack in Python First

```
src/tiny_llm/quantize.py
src/tiny_llm/embedding.py
```

Before writing any GPU code, write the reconstruction in plain `mlx.core`.
This version becomes the reference you trust when a kernel misbehaves.

The starter's `QuantizedWeights.from_mlx_layer` already loads the packed
fields; reuse it rather than writing another loader. Implement these
functions:

- `dequantize_weights` in `quantize.py`. Unpack each `uint32` with shifts and
  masks, repeat each group's scale and bias across its 128 values, and
  compute `q * scale + bias`. Do not call `mx.dequantize`; the point is to
  write the unpacking yourself.
- `QuantizedEmbedding.__init__`, `QuantizedEmbedding.__call__`, and
  `QuantizedEmbedding.as_linear` in `embedding.py`.

The token embedding table is quantized too, and it is large: `vocab_size`
rows of `hidden_size` values. A lookup should only unpack the rows it needs.
`embedding(input_ids)` gathers the packed words, scales, and biases for the
requested token ids, then calls `dequantize_weights` on just those rows. The
ids may arrive as `uint32` values from the sampler, so accept them as well as
`int32`.

Qwen3 ties its output head to the embedding table: the same matrix that maps
token ids to vectors also maps the final hidden state back to vocabulary
logits. `embedding.as_linear(h)` is that output projection. Implement it as
`quantized_linear(h, weight)` so it reuses the quantized-matmul path instead
of expanding the whole `vocab_size × hidden_size` table. It starts working
once Task 3 is done.

Also implement two thin wrappers in `quantize.py`:

- `quantized_linear(x, w, bias=None)` takes the same arguments as Week 1's
  `linear`. On the course path, it chooses a kernel from the number of rows,
  as described in Task 3.
- `mlx_quantized_linear(x, w, bias=None)` computes the same result with
  `mx.quantized_matmul(..., transpose=True)`. This is a control, not your
  implementation: the Task 4 model test builds a second copy of your model
  that routes through it, then compares the two. `quantized_linear` should
  call it only when `w.use_mlx_quantized_linear` is true.

The Task 1 tests check `dequantize_weights` and the embedding lookup against
`mx.dequantize`:

```bash
pdm run test --week 2 --day 2 -- -k task_1
```

## Task 2: Declare the Operator

```
src/extensions/src/tiny_llm_ext.h
src/extensions/bindings.cpp
src/extensions/src/quantized_matmul.cpp
src/extensions/CMakeLists.txt
```

An MLX extension operator has two parts. A C++ function checks its inputs and
returns a lazy `mx::array` that records "this array will be produced by my
primitive." Later, when MLX evaluates the graph, it calls the primitive's
`eval_gpu` (or `eval_cpu`) to do the work. MLX owns the arrays and the command
encoder; you supply the shape logic and the kernel launch. This is the same
pattern as the `axpby` example in the MLX extension guide.

Most of the plumbing is already in place. Read the Day 2
`quantized_matmul(...)` declaration and the `QuantizedMatmul` class in
`tiny_llm_ext.h`, and check that the existing `m.def("quantized_matmul", ...)`
binding and the `quantized_matmul.cpp` entry in `CMakeLists.txt` match it.
Do not add a second binding or a duplicate source entry. Keep everything in
the `tiny_llm_ext` namespace.

Then fill in `quantized_matmul.cpp`:

- `tiny_llm_ext::quantized_matmul(...)` in `quantized_matmul.cpp` checks the
  inputs and returns the lazy output array of shape `(M, K)`. Require
  `group_size == 128` and `bits == 4`, BF16 scales, biases, and activations
  that all share one dtype, and shapes that agree with each other.
- `QuantizedMatmul::eval_cpu(...)` in `quantized_matmul.cpp` should throw a
  clear error. This operator runs only on the GPU; there is no CPU fallback
  to keep in sync.

On the Python side, `quantized_matmul` in `quantize.py` flattens the leading
dimensions of `a` into `M`, calls `mx.contiguous` on each input, calls the
extension, and reshapes the result back. Metal receives raw buffers, not
strides, so the extension can only index correctly if the data is laid out
row by row. Make that a rule at the boundary: call `mx.contiguous` once in
Python, and check it again in C++ before encoding the kernel.

Build to catch signature and binding mismatches early:

```bash
pdm run build-ext
pdm run test --week 2 --day 2 -- -k task_1
```

## Before You Write Metal: How to Debug a Kernel

You will not write a C++ CPU version of this operator, so GPU bugs need a
plan. Use three levels, each checked against the one before it:

1. **The equation in Python** (`dequantize_weights` plus a normal matmul, or
   `mx.quantized_matmul`). This defines the right answer.
2. **A simple Metal kernel** where one thread computes one output. It is slow,
   but you can read it line by line against the equation.
3. **The fast Metal kernel**, using SIMD groups and wider loads. Compare it
   with the simple kernel, not with whole-model text output, which is too far
   from the bug to tell you anything.

Pick inputs whose answers you can predict: zeros, ones, a ramp, an identity
matrix, and a fixed random seed. Try one shape that fits the kernel's tiles
exactly and one with a leftover tail, such as 8 and then 10 rows for an
8-row tile.

MLX is lazy, so an error may not appear where it happened. Force evaluation
right after the operator you are testing:

```python
expected = python_reference(*inputs)
actual = metal_operator(*inputs)
mx.eval(expected, actual)

assert actual.shape == expected.shape
assert actual.dtype == mx.bfloat16
assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()
```

When something fails, check the boundary before the arithmetic: rank, shape,
dtype, and contiguity in Python and C++, and whether each buffer index in the
encoder matches the Metal function signature. The kind of failure narrows
the search:

- **Pipeline creation fails:** the kernel name, its template instantiation,
  or the Metal source is wrong.
- **The GPU faults or reads garbage:** a grid size, bounds check, stride, or
  buffer binding is wrong.
- **The result is finite but wrong:** indexing, reduction, masking,
  unpacking, or the accumulator update is wrong.

For a wrong number, make the kernel simpler until it agrees: one output per
thread, no cooperative loads. Then compare an intermediate, such as one
unpacked group or one partial dot product, by writing it to a small debug
output buffer. Printing from thousands of GPU threads is rarely readable. Put
optimizations back one at a time, rerunning the aligned and tail shapes after
each.

## Task 3: Write the Simple Metal Kernel

```
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
src/tiny_llm/quantize.py
```

Today you write one kernel, `quantized_matmul_vanilla_w4a16_g128`, with one
thread per output. It serves every number of activation rows: a one-row
decode step and a 128-row prompt alike. Implement:

- `quantized_matmul_vanilla_w4a16_g128` in `quantized_matmul.metal`,
  instantiated for `bfloat16_t`;
- `QuantizedMatmul::eval_gpu` in `quantized_matmul.cpp`, launching it;
- `quantized_matmul` and `quantized_matmul_vanilla` in `quantize.py`, so the
  kernel can be called directly in tests and comparisons.

### A Quick Map of the GPU

Metal organizes work in four levels. You need all four to read the kernels
below:

- **Thread (lane).** One instruction stream with its own registers.
- **SIMD group.** 32 lanes on Apple GPUs that run in lockstep. Lanes in one
  group can combine values directly with functions like `simd_sum`, without
  going through memory.
- **Threadgroup.** One or more SIMD groups scheduled together on one GPU core.
  They can share `threadgroup` memory and wait for each other with
  `threadgroup_barrier`.
- **Grid.** All the threadgroups in one launch. `dispatchThreadgroups` sets how
  many there are, and the GPU spreads them across its cores.

### Stage 1: The Simple Kernel

Launch a 2D grid with one thread per output `(i, k)`. Each thread walks all
`N` inputs, unpacks eight codes from each `uint32`, applies the group's scale
and bias, and accumulates in `float32`. Check `i < M` and `k < K` before
writing.

This kernel reloads the same activations in every thread and shares no work,
so it is slow. Keep it anyway. It mirrors the loop in the previous section
almost line for line, which makes it the kernel you compare everything else
against. `quantized_matmul_vanilla` should always reach it.

### Connect `eval_gpu`

Follow the `axpby` dispatch pattern:

1. Get the Metal device and command encoder from the stream.
2. Choose the kernel. The binding's `use_simdgroup` flag selects faster
   kernels on later days. Today, launch the simple kernel whenever
   `use_simdgroup` is false, whatever the number of rows.
3. Bind the inputs, the output, and the constants `M`, `N`, and `K`, in the
   order of the kernel's buffer indices.
4. Size a 2D grid with one thread per output and call `dispatchThreadgroups`.

On the Python side, `quantized_matmul_vanilla` calls `quantized_matmul` with
`use_simdgroup=False`, and `quantized_linear` calls `quantized_matmul` with
`use_simdgroup` set from the weight's `use_simdgroup_matmul` field, which is
false today. Day 3 adds a separate branch for small row counts.

Then run the GPU tests:

```bash
pdm run build-ext
pdm run build-ext-ref
pdm run test --week 2 --day 2 -- -k task_3
```

These compare your simple kernel with `mx.quantized_matmul` on a 128-row
identity input and a 3-row random input.

## Task 4: Use It in the Model

```
src/tiny_llm/qwen3_week2.py
```

Now make the cached model use the new operator. Update
`Qwen3ModelWeek2.__init__`, `Qwen3MultiHeadAttention.__call__`,
`Qwen3MLP.__call__`, and `Qwen3ModelWeek2.__call__`.

Add the packed path next to the dense one; do not replace it. The model takes
a `checkpoint` name, and `WEEK2_CHECKPOINT_FEATURES` in `qwen3_week2.py`
records what each checkpoint turns on. `quantized_weights` is false for
`kv-cache` and true from `packed-weights` onward. Keep the dense BF16 weights,
embedding, and `linear` calls for `kv-cache`, so it stays a fair control. When
`quantized_weights` is true:

- Load every attention projection (`wq`, `wk`, `wv`, `wo`) and MLP projection
  (`w_gate`, `w_up`, `w_down`) with `QuantizedWeights.from_mlx_layer(...)`
  instead of expanding it to BF16, and call `quantized_linear(x, w)` where you
  used to call `linear(x, w)`.
- Load `embed_tokens` the same way and wrap it in your `QuantizedEmbedding`.
- If the model has a separate `lm_head` (that is, the embeddings are not
  tied), keep it as `QuantizedWeights` too and apply it with
  `quantized_linear`. It is a projection, not a lookup.
- Pass the constructor's `use_mlx_quantized_linear` flag through to
  `from_mlx_layer`. The test builds a control model with this flag set and
  expects its projections to go through `mlx_quantized_linear`.
- Keep `group_size` and `bits` from each loaded layer, and let the extension
  reject anything other than 128 and 4.
- For `packed-weights`, load the weights with `use_simdgroup_matvec=False`
  and `use_simdgroup_matmul=False`, so every projection goes to the simple
  kernel. The checkpoint's `simdgroup_matvec` and `simdgroup_matmul` features
  turn these on later.

Leave the Week 1 model alone. Its layers still take plain `mx.array` weights.

If the model outputs `nan` or nonsense, check dtypes first. Scales, biases,
and activations must all be BF16, and the output must be BF16 too.

Run the whole Day 2 gate, then generate text with your model:

```bash
pdm run test --week 2 --day 2

pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint packed-weights --model qwen3-0.6b --max-tokens 16
```

## Check That the Model Really Uses Your Kernel

Passing tests is not quite enough. The tests compare outputs and call your
operators directly, but they do not check which path the model takes
internally. Make sure that both a decode step and a long prompt go
`quantized_linear` → `quantized_matmul` → your primitive → the simple kernel.
An easy way to check is to put a temporary exception in `quantized_matmul`
and confirm that the `main` command above hits it.

Then measure against Day 1, keeping the model, prompt length, output length,
and logit mode identical. Putting both checkpoints in one benchmark run lets
the runner alternate them, so a slow minute on your machine does not land on
only one side:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 --repeats 2 \
  --variant week2-kv-cache --variant week2-packed-weights --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day2-packed.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case kv-cache:decode:128 --case packed-weights:decode:128 \
  --warmup 4 --iterations 12 \
  --json-output week2-day2-attribution.json
```

The profiler replays one decode step that ends at 128 cached tokens through
the public operators each checkpoint selects, as described in
[Day 1](./week2-01-kv-cache.md#find-where-the-time-goes). Both cases share the
same context and logit setting; the only thing that changes is the
packed-weight path, for the projections and the embedding.

Then answer three questions from your own runs. Did packing change the
replayed projections group, and by how much? Did whole-request decode change?
Did peak allocation fall, now that the weights stay packed? Three outcomes
are common, and each tells you something:

- **The projections group fell and the request got faster.** The mechanism
  you wrote down survived a test that could have disproved it. Note how much
  of the replay change showed up in the request; it is rarely all of it.
- **The group fell but the request did not move.** Something else was the
  limit: another group, the CPU, or prefill rather than decode. Check the
  prefill and decode split in the benchmark before you go further.
- **The request got slower.** Keep the result. Check that the model still
  calls the new path, then look for the group that grew. It can happen
  today: one thread per output reads fewer bytes, but on a single decode row
  it leaves most of the GPU idle.

You will ask the same three questions at the end of every later day. Keep the
`packed-weights` result: it is [Day 3](./week2-03-decode-matvec.md)'s control.

## If You Get Stuck

If you need to move on before your kernels work, you can implement
`quantized_linear` with `mx.quantized_matmul`, in the same way as
`mlx_quantized_linear`, and leave the rest of your model unchanged. That
swaps out one operator and keeps your cache and model wiring. It is not the
same as `--solution mlx`, which runs a separate, complete MLX model and
exercises none of your code.

{{#include copyright.md}}
