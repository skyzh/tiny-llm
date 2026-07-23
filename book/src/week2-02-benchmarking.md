# 🚧 Week 2 Day 2: Benchmark and Profile

> 🚧 This chapter is under review and may change.

Optimization starts with a trustworthy comparison. In this chapter, we measure
prefill and decode separately, synchronize MLX's lazy execution inside every
timed iteration, and record a baseline that later changes must beat.

## Prefill and Decode Are Different Workloads

Prefill processes many prompt tokens at once, so its matrix multiplications have
a larger row dimension. Decode usually processes one token per request and is
dominated by repeatedly reading quantized weights. A change can improve one
phase while hurting the other, so `bench.py` reports both:

- prefill tokens per second: prompt tokens divided by prefill time;
- decode tokens per second: generated tokens after the first token divided by
  decode time.

The first generated token belongs to prefill. Excluding it from decode prevents
prompt length from distorting the decode number.

Choose the prefill workload before comparing implementations. Prompt scoring
needs logits for every position, while serving needs only the final prompt
logit. Use `--prefill-logits all` for the former and
`--prefill-logits last` for the latter. The runner applies the choice to the
course model and MLX alike. Never compare a final-row course run with an
all-row MLX run. Cached decode has `L = 1`, so `logits_to_keep=1` removes no
decode work.

Both sides of the Week 2 comparison use a KV cache: prefill the prompt
once, then pass only the newly generated token on each decode step. Comparing a
cached MLX baseline with a course model that recomputes the full prefix would
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

From Day 3 onward, the course extensions are GPU-only. A second C++ CPU
implementation would duplicate the equation without exercising the dispatch,
memory, or synchronization behavior that makes a Metal kernel fail. Use this
three-level validation ladder instead:

1. Write the equation in readable Python/MLX. This is the semantic oracle.
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
The Week 2 operator ladder compares the readable implementation, the optimized
course implementation, and MLX at the selected model's real tensor shapes:

```bash
pdm run bench-week2-operators --model qwen3-0.6b --context 128
```

For the measurements quoted later in this week, the machine was an M4 Pro with
a 20-core GPU and 64 GB of memory. Each operator used 20 warmup iterations and
100 synchronized timed iterations in each of three fresh processes. The
reported result is the median process-level speedup. The matched end-to-end
commands used two complete warmups and three fresh measured runs.

## Attribute Time With a Real GPU Profile

An end-to-end benchmark tells us whether a checkpoint improved. A GPU profile
tells us what to optimize next. Keep those jobs separate:

1. Run the acceptance benchmark outside a trace and save its JSON output.
2. Capture a short, representative request with Instruments or `xctrace`.
3. Rank shader pipelines by total GPU duration, then inspect dispatch count and
   duration per dispatch.
4. Optimize the highest measured cost whose scaling matches the target
   workload.
5. Re-run the isolated operator, end-to-end benchmark, and profile before
   choosing the next chapter.

The command-line profiler is available with Xcode:

```bash
xcrun xctrace list templates

xcrun xctrace record \
  --template /path/to/TinyLLMMetal.tracetemplate \
  --output /tmp/week2.trace \
  --launch -- pdm run bench-week2-operators \
    --model qwen3-4b --context 32 --prefill-projection k

xcrun xctrace export --input /tmp/week2.trace --toc
```

The stock Metal System Trace template is useful for command-buffer and queue
behavior. To see shader names, timelines, and counters, save a custom
Instruments template containing the Metal Shader Timeline and the relevant
counter set, then pass that template to `xctrace`. Xcode GPU Capture remains
the better tool for inspecting a single dispatch in detail.

Do not use trace-instrumented wall time as a throughput result: capture adds
overhead. Record at least the model and tensor shape, pipeline or shader name,
total GPU duration, dispatch count, duration per dispatch, and share of the
captured GPU interval. If the trace does not identify a dominant kernel, do not
invent one from the source code; first shorten the workload, add signposts, or
use an operator ablation that makes attribution unambiguous.

## Record a Matched Baseline

Use the same model, prompt length, output length, device, and warmup count for
your implementation and the MLX run. Replace `tiny_llm` with `tiny_llm_ref` to
compare against the course reference:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2 \
  --prefill-logits last

pdm run bench --solution mlx --loader week2 --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2 \
  --prefill-logits last
```

Or run the complete cumulative ladder in fresh processes. At this point, only
the Week 1, KV-cache, and MLX rows are course prerequisites; later rows become
meaningful as you complete their chapters.

```bash
pdm run bench-week2-progression --offline --repeats 3 \
  --model qwen3-0.6b --input-len 128 --output-len 65 --warmup 2 \
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
reference prefill throughput / MLX prefill throughput >= 0.70
reference decode throughput / MLX decode throughput >= 0.70
```

Reaching 70% is the acceptance threshold, not a promise that every educational
kernel individually matches its MLX counterpart. MLX is the comparison
baseline; the Week 2 solution must reach both targets with course-owned
operator implementations. If either ratio misses, the next chapter starts
from the new benchmark and profile rather than a predetermined optimization.

{{#include copyright.md}}
