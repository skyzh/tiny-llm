# 🚧 Week 2 Day 2: Benchmark and Profile

> 🚧 This chapter is under review and may change.

Optimization starts with a trustworthy comparison. In this chapter, we measure
prefill and decode separately, synchronize MLX's lazy execution inside every
timed iteration, and record a baseline that later changes must beat.

## Prefill and Decode Are Different Workloads

Prefill processes many prompt tokens at once, so its matrix multiplications have
a larger row dimension. Decode usually processes one token per request and is
dominated by repeatedly reading quantized weights. A change can improve one
phase while hurting the other, so `benches/bench.py` reports both:

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
all-row MLX run. Cached decode has `L = 1`, so `logits_to_keep=1` removes no
decode work.

Both sides of the Week 2 comparison use a KV cache: prefill the prompt
once, then pass only the newly generated token on each decode step. Comparing a
cached MLX baseline with your solution recomputing the full prefix would
measure two different algorithms and make the kernel target meaningless. Day 1
already produced the cached readable model used as this week's starting point.

## Synchronize Lazy Work

MLX builds lazy computation graphs. Timing only the Python call measures graph
construction, not GPU execution. Every timed iteration must evaluate the output:

```python
start = perf_counter()
output = function()
mx.eval(output)
elapsed = perf_counter() - start
```

The benchmark must also release request-owned caches after warmups and timed
runs so a later sample does not inherit allocator state:

```bash
pdm run test --week 2 --day 2
```

## Debug Metal Without a CPU Twin

From Day 3 onward, the extensions in your solution are GPU-only. A second C++
CPU implementation would duplicate the equation without exercising the
dispatch, memory, or synchronization behavior that makes a Metal kernel fail.
Use this three-level validation ladder instead:

1. Write the equation in readable Python with `mlx.core`. This is the semantic
   oracle.
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
kernels agree with the readable oracle.

The isolated benchmarks use the same rule. Evaluate input setup
before invoking the benchmark fixture so setup does not leak into the result.
The Week 2 operator ladder compares the readable equation, the optimized kernel
in your solution, and MLX at the selected model's real tensor shapes:

```bash
pdm run bench-week2-operators --model qwen3-4b --context 128
```

Choose enough warmup iterations to exclude compilation, synchronize every
timed iteration, and repeat the run in fresh processes. Report the median with
the exact hardware, dependency versions, model, and tensor shapes. The
[performance appendix](./appendix-performance.md) applies this protocol to the
reference-solution checkpoints and keeps the resulting machine-specific numbers
in one place.

To rank complete model work without requiring a GUI, replay the actual
reference-solution kernel groups at Qwen3-4B shapes and dispatch counts:

```bash
pdm run profile-week2-kernels --model qwen3-4b \
  --warmup 5 --iterations 15 \
  --json-output week2-kernel-profile.json
```

The projection group preserves the transformer dependency order, including the
attention projections before the output projection and the MLP after the
attention residual. This matters for occupancy: making every layer independent
would let unrelated work hide an under-filled kernel and produce a false
Split-K conclusion. The runner synchronizes once per group and normalizes the
group medians into an attribution profile.

The resulting shares are not a throughput benchmark. Group boundaries force
materialization that a complete lazy graph may fuse, while a capture adds its
own overhead. Use the profile to rank kernel groups, then require the ordinary
fresh-process model benchmark to confirm the change. The
[performance appendix](./appendix-performance.md#the-kernel-profile-that-selects-each-chapter)
contains the reference-solution operator-attribution chart and raw-result path.

## Inspect the Metal Pipeline

An end-to-end benchmark tells us whether a checkpoint improved. A GPU profile
tells us what to optimize next. The measurements answer different questions:

| Question | Measurement |
|---|---|
| Did the complete model improve? | Fresh-process throughput benchmark |
| Which operator family dominates? | Synchronized kernel-group attribution |
| Which shader, function, and source line is expensive? | Metal Pipeline Statistics and Shader Cost Graph |

The operator-attribution chart is not a flame graph. On M3 and newer Macs,
Xcode's
[Shader Cost Graph](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-shader-cost-graph-a17-m3)
is the flame graph: it ranks shader function calls and connects them to
weighted source lines.
[Pipeline Statistics](https://developer.apple.com/documentation/xcode/analyzing-draw-command-and-compute-dispatch-performance-with-pipeline-statistics)
separates GPU time into ALU, memory, control-flow, and synchronization
activity.

Build the reference extension with source and line tables, then capture one
Qwen3-4B projection at its real shape:

```bash
CMAKE_ARGS="-DMLX_METAL_DEBUG=ON" pdm run build-ext-ref

MTL_CAPTURE_ENABLED=1 pdm run capture-week2-shader \
  --projection q --rows 1 \
  --output /tmp/week2-q-projection.gputrace

open /tmp/week2-q-projection.gputrace
```

The capture uses synthetic buffers with the real `M=1`, `K=2560`, `N=4096`
Qwen3-4B shape. This keeps the trace small enough to replay without embedding
all model weights; the dispatched reference-solution kernel and its schedule
are unchanged. The warmup and input materialization happen before capture, so
the trace contains steady-state GPU work rather than first-use compilation.

In Xcode:

1. Profile the GPU trace and select the
   `quantized_matvec_x4_fast_w4a16_g128_bf16` compute pipeline.
2. Open Pipeline Statistics. Record GPU time and the ALU, memory, control-flow,
   and synchronization breakdown.
3. Open **Performance > Shaders**. Use the Shader Cost Graph to find the
   highest-cost function call, then select it to jump to the weighted Metal
   source lines.
4. Record the dominant line's cost, executed-instruction count, divergence,
   and instruction categories such as load/store, conversion, bit
   manipulation, and arithmetic.

Missing source lines mean the extension was not rebuilt with
`MLX_METAL_DEBUG`. Missing counter samples mean the selected profiler is not
supported on that OS, Xcode, or GPU combination; neither case justifies an
ALU- or bandwidth-bound claim.

Use the result in this order:

1. Run the acceptance benchmark outside a trace and save its JSON output.
2. Use operator attribution to select a kernel family.
3. Capture that kernel, rank pipeline and function cost, and inspect the
   dominant source lines.
4. Optimize the highest measured cost whose scaling matches the target
   workload.
5. Re-run the isolated operator, end-to-end benchmark, and profile before
   choosing the next chapter.

For a longer request, Instruments can complement the single-dispatch capture:

```bash
xcrun xctrace list templates

xcrun xctrace record \
  --template /path/to/TinyLLMMetal.tracetemplate \
  --output /tmp/week2.trace \
  --launch -- pdm run bench-week2-operators \
    --model qwen3-4b --context 32 --prefill-projection k

xcrun xctrace export --input /tmp/week2.trace --toc
```

The stock Metal System Trace is useful for command buffers, queues, and GPU
intervals. A compatible Metal Shader Timeline or counter template can rank
pipelines over the longer request. Use the source-enabled GPU trace for the
per-function flame graph and per-line activity breakdown.

Do not use trace-instrumented wall time as a throughput result: capture adds
overhead and profiling may serialize commands. Record at least the tensor
shape, pipeline name, GPU time, dispatch count, Pipeline Statistics activity,
highest-cost function, and highest-cost source line. If the trace does not
identify a dominant cost, do not invent one from the source code. Shorten the
workload or return to the dependency-aware operator attribution, then check
that its attributed total and the complete-model phase time move in the same
direction.

## Record a Matched Baseline

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

Or run the complete cumulative ladder in fresh processes. At this point, only
the Week 1, KV-cache, and MLX rows are course prerequisites; later rows become
meaningful as you complete their chapters.

```bash
pdm run bench-week2-progression --offline --repeats 3 \
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

## Acceptance Target

The Week 2 targets are:

```plain
your solution's prefill throughput / MLX prefill throughput >= 0.80
your solution's decode throughput / MLX decode throughput >= 0.80
```

The prefill ratio is also 0.80; both ratios use Qwen3-4B, a 128-token prompt,
128 timed decode steps, and last-row logits. `--output-len 129` includes the
first token produced by prefill. Reaching 80% is the acceptance threshold, not
a promise that every educational kernel individually matches its MLX
counterpart. MLX is the comparison baseline; your solution must reach both
targets with its own operator implementations. If either ratio
misses, the next chapter starts from the new benchmark and profile rather than
a predetermined optimization.

Keep a 2K context run in the report as a stress diagnostic. It is useful for
showing when attention overtakes fixed-shape projections, but changing context
also changes the problem. Do not move the acceptance shape after seeing a
result.

{{#include copyright.md}}
