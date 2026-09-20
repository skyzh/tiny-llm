# 🚧 Packed W4: Read Less for Each Decode Token

The incoming `kv-cache` checkpoint already processes one new token at a time.
It still uses dense BF16 projections. When you inspect a decode projection,
each output row takes a dot product between the new hidden state and a weight
row. Keeping that row packed until the multiply reduces the weight bytes the
operator needs to read.

The starter supplies `QuantizedWeights` and its `from_mlx_layer` loader. You own
the selected-row embedding lookup, quantized-linear wrappers, native primitive,
Metal schedules, and model integration. Begin at the Python boundary:

```bash
pdm run build-ext
pdm run test --week 2 --day 3 -- -k task_1
```

The build should expose the supplied extension stubs. The tests will fail until
you implement the embedding and unpacking work. Do not expect the complete
`quantized-matvec` product to run yet.

## Decode One Stored Weight

Use `W` for a logical weight matrix with shape `N, K`, where `N` is the output
width and `K` is the input width. Eight unsigned four-bit codes fit in a
`uint32`. With group size 128, the representation is:

| Array | Shape | Dtype |
|---|---|---|
| Packed weight codes | `N, K / 8` | `uint32` |
| Scales | `N, K / 128` | BF16 |
| Biases | `N, K / 128` | BF16 |

For input coordinate `k`, extract its code with a shift and `0xF` mask. Read the
scale and bias for group `k // 128`, and reconstruct:

$$
\widehat{w}_{n,k} = q_{n,k}s_{n,\lfloor k/128\rfloor}
                   + b_{n,\lfloor k/128\rfloor}.
$$

Use the stored affine parameters directly, including the sign of the scale.
For example, with scale `-0.25` and bias `1`, code zero reconstructs to `1` and
code four to `0`. Assuming every scale is positive would silently change this
weight row.

Codes occupy half a byte per logical weight; a BF16 scale and bias add four
bytes per group. Thus this representation uses `0.5 + 4/128 = 0.53125` bytes per
weight, excluding container overhead. That is a storage calculation, not a
prediction of request throughput: the model also reads activations and K/V and
executes other operators.

## Keep Both Embedding Uses Packed

Complete `dequantize_weights` and `quantized_linear` in
`src/tiny_llm/quantize.py`, and the `QuantizedEmbedding` methods in
`src/tiny_llm/embedding.py`.

For token lookup, gather only the requested packed rows and their metadata,
then unpack those rows with basic MLX operations. The supplied ownership check
expects your unpacking, rather than a call to `mx.dequantize`. Do not expand the
whole vocabulary table merely to read a few token rows.

The same embedding can serve as the tied output projection through `as_linear`.
That call must use your quantized-linear path. Unlike lookup, it produces one
logit per vocabulary entry; gathering a few rows is not equivalent. Reuse
`QuantizedWeights.from_mlx_layer` for loading rather than creating another
packed-weight representation.

## Bring Up the Native Dot Product

For activations `A` shaped `M, K`, compute `C = A Wᵀ`, shaped `M, N`. Accumulate
each dot product in FP32 and cast its final output to BF16. Keep codes packed in
memory, reconstructing the values needed by the current work inside the kernel.

The existing extension surface is:

- `src/extensions/src/tiny_llm_ext.h`: `quantized_matmul` and its primitive;
- `src/extensions/src/quantized_matmul.cpp`: validation and GPU dispatch;
- `src/extensions/src/quantized_matmul.metal`: the matrix and matvec kernels;
- `src/extensions/bindings.cpp`: the already-registered Python binding.

Replace the fail-closed bodies rather than adding a second API. Validate ranks,
dimensions, dtypes, group size, and bit width before dispatch. Return a lazy MLX
array whose primitive evaluates on the GPU; CPU evaluation can reject this
GPU-only exercise explicitly.

Start with `quantized_matmul_vanilla_w4a16_g128`: one output element and a clear
reduction loop. Then implement `quantized_matvec_x4_fast_w4a16_g128`, sharing dot
product work within SIMD groups for the small-row path. Connect
`QuantizedMatmul::eval_gpu` and the explicit Python comparison wrappers
`quantized_matmul_vanilla` and `quantized_matvec_custom`.

A lane performs its own loads and arithmetic. A SIMD group can combine lane
partials with operations such as `simd_sum`. A threadgroup can hold multiple
SIMD groups and share explicitly synchronized memory. The grid distributes
threadgroups across the output. More groups can expose parallel work, but also
increase resource demand or duplicate reads; treat the launch schedule as an
experiment after the simple result is correct.

At this checkpoint, use the matvec schedule for small activation-row counts and
keep the simple matrix implementation for larger ones. The next chapter
replaces the latter. Rebuild and compare GPU results:

```bash
pdm run build-ext
pdm run test --week 2 --day 3 -- -k gpu
```

When a result is wrong, first check the packed index, group index, and output
shape on a small input. Force evaluation near the failing call so a deferred
GPU error does not appear to belong to a later operator. Check a tail shape as
well as an aligned shape before optimizing further.

## Put It Back into the Cached Model

Wire packed weights into `Qwen3ModelWeek2` at `quantized-matvec`, including the
embedding's output projection. Keep RMSNorm, RoPE, and SwiGLU on the readable
Week 1 equations, and keep the same capacity-backed cache. The required GPU
quantized path calls your extension rather than `mx.quantized_matmul`.

```bash
pdm run test --week 2 --day 3
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-0.6b
```

Compare against `kv-cache`, with identical request settings:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-kv-cache --variant week2-quantized-matvec \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-w4.json

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case kv-cache:decode:128 --case quantized-matvec:decode:128 \
  --warmup 4 --iterations 12 --json-output week2-w4-attribution.json
```

The earlier checkpoint is the dense control; the vanilla matrix wrapper is the
local arithmetic control. Check model correctness separately from throughput:
changing weight representation can change numerical results. Record both phases
before deciding whether the measured request benefits.

Prefill has many rows that could reuse each weight tile. Continue to
[Matrix Prefill](./week2-05-simd-matrix-prefill.md) from `quantized-matvec`, without
implementing the compact primitive kernels yet.

{{#include copyright.md}}
