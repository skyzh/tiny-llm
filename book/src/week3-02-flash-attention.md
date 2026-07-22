# Week 2 Days 4-5: FlashAttention-2

> **Status:** Under review (WIP).

In this chapter, we will implement a small FlashAttention-style Metal kernel
for the Week 2 Qwen3 serving pipeline. The goal is to learn the tiled,
IO-aware algorithm and map both matrix multiplications to Metal's public
`simdgroup_matrix` API. We will not reuse an MLX attention-kernel
implementation.

**📚 Readings**

- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [MLX Extension Development Guide](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Feature Set Tables](https://developer.apple.com/metal/feature-sets/)

## Why FlashAttention?

Standard attention computes:

1. `S = QK^T`
2. `P = softmax(S + mask)`
3. `O = PV`

This path materializes an `L x S` score tensor, and sometimes a separate
probability tensor, in device memory. FlashAttention instead streams K/V tiles
through on-chip memory and combines them with online softmax. Its auxiliary
memory is linear rather than quadratic in sequence length.

For Qwen3-4B, one BF16 score tensor at `L = S = 4096` occupies:

```plain
1 batch × 32 query heads × 4096 × 4096 × 2 bytes = 1 GiB
```

Avoiding that allocation is useful on unified-memory Apple silicon too.
However, lower memory use does not automatically mean lower latency. Two
optimized matrix multiplications can beat a fused kernel whose inner matrix
multiplications are poorly mapped to the GPU.

## Online Softmax Recap

For one query row, split keys and values into tiles `j = 1..T`:

$$
m^{(j)} = \max\left(m^{(j-1)}, \max(s^{(j)})\right)
$$

$$
l^{(j)} = e^{m^{(j-1)} - m^{(j)}} l^{(j-1)} + \sum e^{s^{(j)} - m^{(j)}}
$$

$$
o^{(j)} = e^{m^{(j-1)} - m^{(j)}} o^{(j-1)} + \sum e^{s^{(j)} - m^{(j)}} v^{(j)}
$$

At the end:

$$
o = \frac{o^{(T)}}{l^{(T)}}
$$

The running maximum prevents overflow. The correction factor
`exp(old_max - new_max)` rescales both the previous denominator and the
previous output when a later tile contains a larger score.

## The Qwen3 Target

The optimized course path is deliberately specialized instead of pretending
one kernel is ideal for every model:

```plain
Qwen3-4B attention
query heads: 32
KV heads:     8
head_dim:     128
dtype:        bfloat16
```

The supported dense Qwen3 models used by this course have `head_dim = 128`,
including Qwen3-4B. Do not derive this value as `hidden_size / num_heads`:
Qwen3 stores `head_dim` explicitly and its attention projection width can
differ from `hidden_size`.

### Week 2 Dtype Contract

Yes: Week 2 uses BF16 by default. `Qwen3ModelWeek2` sets its model precision to
`mx.bfloat16`, and the quantized projections produce BF16 Q, K, and V. The
FlashAttention path must not upcast those complete tensors to FP32.

Use mixed precision at these boundaries:

| Value | Dtype | Reason |
| --- | --- | --- |
| Q, K, V in device memory | BF16 | matches the Week 2 model and halves traffic versus FP32 |
| Q/K/V threadgroup tiles | BF16 | preserves the bandwidth and capacity benefit |
| attention scale | FP32 | avoids rounding the scalar before repeated use |
| score and output accumulators | FP32 | protects dot products and online softmax |
| running max and sum | FP32 | numerical stability |
| additive mask inside the extension | FP32 | preserves `-inf` and additive bias behavior |
| final attention output | BF16 | feeds the next Week 2 layer without a full-tensor cast |

“Use BF16” therefore does not mean performing the softmax recurrence in BF16.
It means keeping model-sized tensors BF16 and promoting only register-resident
arithmetic that benefits from FP32. A caller may supply a BF16 additive mask;
the wrapper converts its contiguous broadcasted representation to FP32 at the
extension boundary.

## Task 1: Implement the Python Wrapper

```plain
src/tiny_llm/attention.py
```

Implement `flash_attention(query, key, value, scale=None, mask=None)` using the
same model-facing layout as the earlier attention chapter:

```plain
query: B..., H_q, L, E
key:   B..., H,   S, E
value: B..., H,   S, E
out:   B..., H_q, L, E
```

Flatten batch and head dimensions before calling C++, then restore the original
layout. Make Q, K, and V contiguous, but preserve their BF16 dtype. Keep the
scalar `factor` in FP32 rather than casting it to the query dtype.

Pass a small integer mask mode to the extension:

| Mode | Meaning | Mask buffer |
| ---: | --- | --- |
| 0 | no mask | one-element placeholder |
| 1 | causal | one-element placeholder |
| 2 | additive mask | contiguous broadcasted `(N, L, S)` FP32 array |

Do not construct a dense all-zero or causal mask. Doing so reintroduces the
quadratic allocation that the fused kernel is intended to avoid.

## Task 2: Implement the CPU Reference

```plain
src/extensions/src/tiny_llm_ext.h
src/extensions/bindings.cpp
src/extensions/src/flash_attention.cpp
src/extensions/CMakeLists.txt
```

Add the MLX primitive, binding, and CPU evaluator. The readable CPU path uses
FP32 and tiled online softmax with `Br = 32` and `Bc = 32`. It is a correctness
reference, not the Week 2 model path. The intentional CPU exception keeps the
algorithm easy to inspect; the required model-facing GPU path is BF16.

Map grouped-query heads with:

```plain
q_kv_ratio = num_heads / num_kv_heads
kv_head = query_head / q_kv_ratio
```

For causal attention, a key is visible when:

```plain
key_index <= query_index + (S - L)
```

The `S - L` offset is required when Q contains only the new tokens but K/V also
contain cached tokens. Skip a K/V tile entirely when all of its keys are in the
future.

## Task 3: Implement the Metal Kernels

```plain
src/extensions/src/flash_attention.metal
src/extensions/src/flash_attention.cpp
src/extensions/CMakeLists.txt
```

### From the Scalar Kernel to the SIMD-Matrix Kernel

The original diagrams described the FP32 scalar implementation. The Qwen3 path
now uses a separate BF16 specialization:

```plain
CURRENT FP32 FALLBACK                  QWEN3 BF16 FAST PATH

Q tile       16 x 128                  64 x 128
K/V tile     32 x 128                  32 x 128
threadgroup  16 SIMD groups            8 SIMD groups
threads      16 x 32 = 512             8 x 32 = 256

QK           serial 128-element dots   8 x 8 matrix fragments
softmax      FP32                      FP32
PV           scalar loops              8 x 8 matrix fragments
output       FP32                      BF16

              variable D               fixed D = 128
                   |                         |
                   +---- correctness --------+---- Week 2 model
```

The fallback remains useful for FP32 tests and smaller head dimensions. Qwen3
uses the right-hand path so the hot loop has no dynamic-D or dtype branches.

### Dispatch Grid and Thread Ownership

The host launches a three-dimensional grid. One threadgroup owns one query
block for one query head in one batch:

```plain
grid = (ceil(L / 64), Hq, B)       threadgroup = (32 lanes, 8 SIMD groups, 1)

grid.x: query block

       qb=0           qb=1                         qb=ceil(L/64)-1
    +-----------+  +-----------+                  +-----------+
Q   | rows 0:64 |  | rows 64:128|       ...       | tail rows |
    +-----------+  +-----------+                  +-----------+

grid.y: query head qh = 0 .. Hq-1
grid.z: batch      b  = 0 .. B-1

TG(qb, qh, b)
    |
    +-- writes O[b, qh, qb*64 : min((qb+1)*64, L), 0:128]
    |
    +-- reads  K/V[b, kv_head, 0:S, 0:128]
                    ^
                    +-- kv_head = qh / (Hq / Hkv)
```

Inside the threadgroup, each SIMD group owns eight of the 64 query rows:

```plain
threadgroup: 8 SIMD groups x 32 lanes = 256 threads

SIMD group       query rows in this tile       output owned
    0                    0 .. 7                 O[ 0: 8, :]
    1                    8 .. 15                O[ 8:16, :]
    2                   16 .. 23                O[16:24, :]
    3                   24 .. 31                O[24:32, :]
    4                   32 .. 39                O[32:40, :]
    5                   40 .. 47                O[40:48, :]
    6                   48 .. 55                O[48:56, :]
    7                   56 .. 63                O[56:64, :]

One 8 x 8 fragment is distributed across 32 lanes.
Each lane holds exactly two fragment elements.
```

### Matrix-Fragment Dataflow

For one SIMD group, QK turns one `8 × 128` query slice into four FP32 score
fragments:

```plain
                         D = 128 = 16 fragments

Q rows [8 x 128]     [ Q0 ][ Q1 ] ... [ Q15 ]
                         |     |            |
                         |     | 16 MMA     |
                         v     v  steps     v
K^T [128 x 32]       +------+------+------+------+
                     | K*,0 | K*,1 | K*,2 | K*,3 |   four columns cover 32 keys
                     +------+------+------+------+
                         |      |      |      |
                         v      v      v      v
scores [8 x 32]      [ S0 ][ S1 ][ S2 ][ S3 ]     four FP32 8 x 8 fragments
```

After online softmax, PV turns those four probability fragments into sixteen
FP32 output fragments:

```plain
P [8 x 32]           [ P0 ][ P1 ][ P2 ][ P3 ]
                         \      \      \      \
                          \      4 MMA steps per output fragment
                           v      v      v      v
V [32 x 128]         [ V*,0 ][ V*,1 ] ... [ V*,15 ]
                           |       |              |
                           v       v              v
O [8 x 128]          [ O0  ][ O1  ] ... [ O15  ]  sixteen FP32 8 x 8 fragments
```

The matrix operands are BF16. Scores, row maxima, row sums, and output
accumulators remain FP32; only the final normalized output is converted to BF16.

### Threadgroup Memory and Registers

Q stays resident while K and V take turns using the same allocation:

```plain
THREADGROUP MEMORY                                      total = 24.75 KiB

+----------------------------------------------------+
| q_tile: BF16 [64 x 130]                 16.25 KiB  |  loaded once
| 128 values + 2 padding values per row              |
+----------------------------------------------------+

+----------------------------------------------------+
| kv_tile: BF16 [128 x 34]                 8.50 KiB  |  reused every tile
|                                                    |
| K phase: transposed K^T [128 x (32 + 2 padding)]   |
| V phase: row-major V  [32 x (128 + 2 padding)]     |
+----------------------------------------------------+

REGISTERS, PER SIMD GROUP

  scores/probabilities:  4 x 8x8 fragments
  output accumulator:   16 x 8x8 FP32 fragments
  online softmax:        running max + running sum for each row
```

The two padding elements give BF16 rows an odd stride in 32-bit memory banks,
reducing systematic bank conflicts during fragment loads.

### Per-Tile Sequence

The barriers line up with each change in ownership of shared memory:

```plain
once per query tile

device Q --load--> q_tile --B--+
                              |
for each K/V tile j            v

device K_j --load + transpose--> kv_tile as K^T --B--> QK
                                                        |
                                                        v
                                           scale + mask + softmax
                                                        |
                         running max, sum, O <----------+
                                                        |
                                                        B
                                                        |
device V_j --load--------------> kv_tile as V   <-------+
                                         |
                                         B
                                         |
                                         v
                                  P @ V, rescale O
                                         |
                                         B
                                         |
                                         +----> next K/V tile

after final tile:  O / running_sum --convert--> BF16 output

B = threadgroup barrier
```

Causal mode bounds the loop before it starts processing tiles:

```plain
last visible key = last query row in block + (S - L)

K/V tiles:  [ 0 ][ 1 ][ 2 ] ... [ limit-1 ] | [ limit ][ limit+1 ] ...
             <---------- process ----------> | <------- skip entirely ------>
```

For `L = S`, this skips approximately half of the matrix work.

### Implemented Optimization Inventory

The following table is the complete inventory of performance choices in the
course implementation. These are present in the Python wrapper, C++ dispatch,
or Metal shader; they are not suggestions borrowed from the later Steel
comparison.

| Layer | Implemented optimization | What it saves or improves |
| --- | --- | --- |
| Python | Preserve BF16 Q/K/V and make only contiguous views | avoids three full-tensor FP32 conversions and keeps device traffic at two bytes per element |
| Python | Represent no-mask and causal mode with an integer plus a one-element placeholder | avoids allocating and reading an `N × L × S` mask; only a real additive mask is broadcast and materialized |
| C++ | Dispatch BF16 `D=128` directly to a dedicated kernel | removes dynamic head-dimension loops and dtype branches from the Qwen3 hot path |
| C++ | Use a three-dimensional grid over query block, query head, and batch | gives each threadgroup one independent output tile without flattening/division work in the tile loop |
| C++/Metal | Map a query head to its KV head with integer GQA indexing | reads the original eight Qwen3-4B KV heads instead of physically repeating K/V for all 32 query heads |
| Metal | Use a `64 × 32` compile-time tile with 256 threads | replaces the original 1024-thread launch and gives each of eight SIMD groups ownership of eight query rows |
| Metal | Use 8×8 `simdgroup_matrix` operations for both QKᵀ and PV | replaces serial dot products and per-output-column SIMD reductions with the GPU's matrix path |
| Metal | Cooperatively stage Q once per query tile | reuses each Q element across every streamed K/V tile instead of rereading Q from device memory |
| Metal | Transpose K while cooperatively loading it | produces the row-major `[D, BK]` layout required by QKᵀ without a separate transpose kernel or buffer |
| Metal | Stream K and V through one padded threadgroup allocation | keeps total threadgroup storage near 25 KiB and avoids simultaneously reserving separate K and V tiles |
| Metal | Pad BF16 threadgroup rows by two elements | changes the 32-bit bank stride from an even pattern and reduces systematic bank conflicts for fragment loads |
| Metal | Keep score fragments, output fragments, and online-softmax state in registers | avoids round trips through threadgroup or device memory between QKᵀ, softmax, and PV |
| Metal | Reduce each fragment row with two lane shuffles | uses the known 8×8 fragment lane mapping instead of a threadgroup reduction and synchronization |
| Metal | Use online softmax and rescale the accumulated output when the running maximum changes | fuses exact softmax with PV and never materializes the quadratic score or probability tensor |
| Metal | Convert only the four probability fragments per tile back to BF16 | enables BF16 PV matrix instructions while retaining FP32 scores, softmax state, and output accumulation |
| Metal | Bound the causal K/V tile loop before entering it | skips every tile wholly to the right of the causal frontier; the current kernel still evaluates its runtime element-level causal condition in every processed tile |
| Metal | Use `fast::exp` for register-resident softmax exponentials | selects Metal's lower-latency approximate exponential while the stable max subtraction controls the input range |
| Metal | Zero-fill partial Q/K/V tiles during cooperative loads | lets one kernel handle arbitrary sequence tails without a second launch or out-of-bounds reads |
| Metal | Synchronize only at shared-tile ownership transitions | barriers protect Q/K readiness, K-to-V reuse, V readiness, and V-to-next-K reuse; matrix and softmax register work needs no threadgroup barrier |
| Metal | Divide by the online denominator only at the final BF16 store | avoids normalizing and rewriting the output after every K/V tile |
| Dispatch | Keep the variable-`D` FP32 scalar kernel separate | preserves a readable correctness fallback without adding generality or extra branches to the Qwen3 specialization |

Two details are easy to miss when reading the shader. First, the manual
`matrix_coord` mapping gives every lane exactly two elements of each 8×8
fragment. `row_max` and `row_sum` therefore need only the `xor(1)` and
`xor(8)` exchanges that connect the lanes holding the same row; a reduction
over all 32 lanes would mix different rows. Second, the final barrier in the
K/V loop is required even though PV writes only registers: it prevents an
early SIMD group from overwriting the shared V tile with the next K tile while
another SIMD group is still reading V.

The current implementation deliberately does **not** include function-constant
mask variants, unchecked aligned interior loaders, Q pre-scaling with
`fast::exp2`, FP32 probability/V fragments, overlapped or double-buffered V
loads, GQA sharing within a threadgroup, tile autotuning, or a decode-specific
kernel. Those are the follow-up exercises below. Keeping this boundary explicit
is important: `simdgroup_matrix` alone delivers the largest structural fix,
while these unimplemented scheduling and specialization techniques explain
most of the remaining gap to MLX Steel.

### Do Not Import an MLX Kernel

The extension necessarily uses MLX's public C++ extension and command-encoder
interfaces, but the Metal shader should include only public Metal headers:

```metal
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
```

Do not include MLX Steel headers or instantiate an MLX attention template. The
point of this task is to implement and understand the fragment mapping,
softmax recurrence, storage layout, and synchronization directly.

### Correctness Cases

Test more than aligned self-attention:

- no mask, additive mask, and causal mask;
- BF16 D=128 as the default GPU path and the FP32 fallback separately;
- partial query and key tiles, such as `L=35, S=47`;
- grouped-query attention;
- `L != S` causal offsets.

Assert that the BF16 GPU result is still BF16. A numerically close FP32 result
would hide a model-sized upcast and fail the Week 2 dtype contract.

Evaluate the lazy result before comparing it. A kernel can compile and appear
to run while a lane-coordinate or partial-tile error remains hidden.

### Benchmark the GPU Work, Not Graph Construction

MLX evaluates lazily. Initialize Q/K/V before timing and force every iteration
to complete:

```python
def evaluate_attention(fn):
    result = fn()
    mx.eval(result)
    mx.synchronize()
    return result
```

On an Apple M4 Pro with MLX 0.29.1, single-request Qwen3-4B causal prefill
(`Hq=32`, `Hkv=8`, `D=128`, BF16) measured approximately:

| Tokens | Explicit attention | MLX fused | Course SIMD-matrix kernel |
| ---: | ---: | ---: | ---: |
| 128 | 0.293 ms | 0.205 ms | 0.538 ms |
| 512 | 1.272 ms | 0.569 ms | 1.899 ms |
| 1024 | 4.755 ms | 1.635 ms | 5.418 ms |
| 2048 | 17.084 ms | 5.584 ms | 19.015 ms |
| 4096 | 66.724 ms | 21.181 ms | 72.717 ms |

The standalone educational kernel does not beat the explicit implementation
on this GPU. At 4096 tokens it is about 9% slower, but it avoids the explicit
path's roughly 1 GiB score tensor. That is still a meaningful implementation:
it demonstrates the correct IO-aware algorithm and has bounded scratch memory.

It also shows why production FlashAttention kernels are complex. MLX's fused
implementation remains about 3.4× faster at 2048 tokens because it adds deeper
specialization and scheduling work beyond merely using matrix instructions.
Do not report memory efficiency as a latency speedup.

### Why MLX Steel Is Much Faster

The MLX 0.29.1 result in the table uses its classic Steel attention kernel. It
uses the same public 8×8 `simdgroup_matrix` operation available to this course,
so the difference is not a different attention algorithm or a larger matrix
instruction. It is the accumulation of many smaller implementation choices.

For D=128, MLX 0.29.1 dispatches a compile-time `BQ=32`, `BK=16`, `BD=128`
specialization with four SIMD groups, or 128 threads. Compare its hot path with
the course kernel:

| Detail | Course kernel | MLX 0.29.1 Steel |
| --- | --- | --- |
| Q/K/V storage | BF16 | BF16 |
| Matrix fragments | BF16 operands, FP32 result | BF16 loads converted to FP32 fragments |
| Probability for PV | converted back to BF16 | remains in FP32 fragments |
| Prefill tile | `64 × 32`, 256 threads | `32 × 16`, 128 threads |
| Scale and exponent | scale every score tile, `fast::exp` | pre-scale Q once, `fast::exp2` |
| Mask selection | runtime `mask_mode` branches | Metal function constants |
| Aligned loads | bounds logic remains in the kernel | unrolled unchecked interior path |
| Loop structure | compiler sees constant bounds | explicit compile-time unrolling |
| V scheduling | load V after softmax | issue the V load before softmax, consume it after |

#### Specialize Away Work That Is Not Needed

Steel uses function constants for aligned Q, aligned K, mask presence, causal
mode, and attention sinks. Metal compiles and caches a different pipeline for
each relevant combination. A causal, aligned, no-additive-mask launch therefore
does not execute runtime branches for the unused modes.

It also distinguishes an aligned interior tile from a partial tail. Interior
tiles use unchecked loads and stores; only the final tile pays bounds checks.
The course kernel checks key and query validity inside every score fragment.

#### Keep the Hot Fragments in FP32

Steel stores BF16 Q/K/V in device and threadgroup memory, but converts values to
FP32 when loading matrix fragments. Scores, probabilities, V fragments, and
output accumulators then stay FP32 through both matrix multiplications.

The course kernel converts FP32 probabilities back to BF16 fragments before
PV. That saves registers, but adds conversions and loses precision. The Steel
choice uses more registers, so it must be paired with its smaller tile and
128-thread threadgroup to preserve occupancy.

#### Move Repeated Arithmetic Out of the K/V Loop

Softmax can be evaluated in base 2:

```plain
exp(x) = exp2(x × log2(e))
```

Steel multiplies Q by `scale × log2(e)` once when loading the Q tile. Every QK
result is already in the units required by `fast::exp2`. The course kernel
multiplies every score in every K/V tile by `scale`, then uses `fast::exp`.

Pre-scaling Q is particularly valuable because Q remains resident while many
K/V tiles stream past it.

#### Use Compile-Time Cooperative Loaders

Steel assigns each thread a compile-time rectangular slice and fully unrolls
the aligned load loops. K is transposed cooperatively while entering threadgroup
memory. Its Q, K, and V loaders advance their source pointers rather than
recomputing full indices in the hot loop. Only the tail path executes validity
checks and zero filling.

Padding is chosen together with the vector width and fragment access pattern.
A padding value is not independently “correct”; it must be evaluated with the
loader, element type, bank mapping, and tile shape.

#### Schedule Loads Around Independent Math

In the Steel loop, the V load is issued after QK but before the softmax
reductions and exponentials. V is not consumed until PV, so the GPU and compiler
have independent work that can overlap the memory operation. The course kernel
finishes softmax before beginning its V load.

Steel also uses SIMD-group barriers where only one SIMD group needs ordering,
and threadgroup barriers only when shared K/V storage is being reused. Barrier
scope and placement matter as much as the raw barrier count.

#### Production Dispatch Uses More Than One Kernel

MLX 0.29.1 selects full tiled attention only when the query length is greater
than eight. Short-query decode uses a separate vector kernel, with a two-pass
variant for long K/V sequences. One prefill tile cannot also be the best decode
mapping.

Later MLX revisions add an optional NAX path using 16×16 cooperative tensors
from MetalPerformancePrimitives. That is not the path measured in this chapter,
and it is outside the course rule that only public `simdgroup_matrix` may be
used for this optimization.

The relevant MLX 0.29.1 sources are useful for comparison after implementing
the exercise independently:

- [host dispatch and tile selection](https://github.com/ml-explore/mlx/blob/v0.29.1/mlx/backend/metal/scaled_dot_product_attention.cpp)
- [Steel attention loop](https://github.com/ml-explore/mlx/blob/v0.29.1/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h)
- [cooperative block loaders](https://github.com/ml-explore/mlx/blob/v0.29.1/mlx/backend/metal/kernels/steel/attn/loader.h)
- [8×8 fragment and MMA helpers](https://github.com/ml-explore/mlx/blob/v0.29.1/mlx/backend/metal/kernels/steel/attn/mma.h)

### Follow-Up Performance Exercises

Treat each item as an experiment: change one mechanism, rerun correctness on
partial tiles and masks, and record the synchronized Qwen3-4B benchmark. Do not
copy or include the Steel headers.

#### 1. Compile-Time Mask Variants

Create separate no-mask, causal, and additive-mask pipelines, either as kernel
entry points or Metal function-constant specializations. For causal mode,
compute the first tile that intersects the diagonal and apply element-level
masking only from that tile onward. Earlier tiles are entirely valid.

This is the smallest follow-up because it changes control flow without changing
the fragment layout.

#### 2. Pre-Scale Q and Use Base-2 Exponentials

Multiply Q by `scale × log2(e)` during the one-time Q load and replace the
online-softmax exponentials with `fast::exp2`. Verify both ordinary and cached
causal attention, because additive masks must be converted to the same base-2
units.

Measure this separately from mask specialization so its effect is visible.

#### 3. Keep Probabilities and V Fragments in FP32

Keep BF16 in device/threadgroup memory, but load Q, K, and V into FP32 matrix
fragments. Feed the FP32 score fragments directly into PV instead of converting
probabilities back to BF16.

Then retune BQ/BK: the additional registers may make a smaller threadgroup
faster. Record numerical error as well as latency.

#### 4. Add Aligned Vector Loaders

Split cooperative loading into:

- an aligned interior path using packed, unchecked reads; and
- a safe tail path that zero-fills out-of-range elements.

Precompute per-thread source and destination offsets and increment pointers
between K/V tiles. Start with fully unrolled scalar loads, then test aligned
packed reads where the source and destination layout permits them. Check
generated memory transactions or Metal GPU counters; source-level vector syntax
alone does not guarantee coalesced loads.

#### 5. Pipeline V Loading with Softmax

Issue the V load before row reductions and exponentials, then place the barrier
immediately before PV consumes V. As a harder variant, double-buffer K/V tiles
so tile `j+1` can be loaded while tile `j` is computed.

Double buffering increases threadgroup memory and can lower occupancy, so it is
an optimization only if the measured overlap exceeds that cost.

#### 6. Autotune Tile and Threadgroup Shapes

Benchmark at least these compile-time variants:

```plain
(BQ, BK, SIMD groups) =
    (32, 16, 4)
    (32, 32, 4)
    (64, 16, 8)
    (64, 32, 8)
```

Run every shape at 128, 512, 1024, 2048, and 4096 tokens. Track threadgroup
memory and estimated register pressure alongside time. A tile may win at long
prefill and regress badly at 128 tokens, so dispatching by sequence length can
be better than choosing one global winner.

#### 7. Reuse K/V Across GQA Query Heads

Qwen3-4B has four query heads per KV head. Explore assigning multiple related
query heads to one threadgroup so a staged K/V tile serves more than one Q head.
This reduces K/V traffic but increases Q storage, output registers, and
threadgroup size. It is a difficult exercise because those resource costs can
erase the reuse benefit.

#### 8. Add a Separate Decode Kernel

Dispatch `L <= 8` to a vector-oriented kernel and consider a two-pass reduction
for very long K/V caches. Do not judge a prefill kernel by one-token decode, or
inflate the prefill kernel with decode-specific branches.

The first four exercises stay close to the current code and are the best next
steps for narrowing the Steel gap. Exercises 5–8 introduce scheduling or
dispatch complexity closer to a production implementation.

### Does FlashAttention Make Sense on Metal?

Yes for memory-bounded prefill, but not for exactly the same reasons or with
the same implementation as an NVIDIA kernel.

Apple GPUs have unified memory, SIMD width 32, Metal threadgroup memory, and a
public 8×8 SIMD-group matrix API. They do not expose the same warp-level tensor
core and asynchronous-copy model as H100/H200. Tile size, bank padding,
occupancy, and synchronization must therefore be retuned for Metal.

Apple's feature tables list SIMD-scoped matrix multiply beginning with Apple
GPU family 7, which includes M1-series GPUs. Check the runtime GPU family if the
course is extended beyond Apple silicon rather than assuming this path exists
on every Metal device.

FlashAttention is less compelling for one-token decode. Decode has almost no
query-tile reuse and is closer to a matrix-vector problem; use a dedicated
decode or paged-attention kernel for that phase.

## Task 4: Preserve the Week Boundary

```
src/tiny_llm/attention.py::flash_attention
```

Keep the FlashAttention implementation behind the Week 3 attention interface.
Do **not** edit `qwen3_week2.py`: that would make the runnable Week 2 checkpoint
depend on a later assignment. At this point, test and benchmark the operator in
isolation for representative prefill shapes.

Days 4 and 5 reuse the same online-softmax and tiling ideas while walking a
paged KV cache. That is the Week 3 model integration point. The dense Day 2
kernel remains a correctness oracle and a useful prefill baseline, while the
Week 2 model remains unchanged and independently runnable.

{{#include copyright.md}}
