# 🚧 Week 2 Day 1: Benchmark MLX

> 🚧 This newly introduced chapter is a work in progress.

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

Both sides of the Week 2 comparison must use a KV cache: prefill the prompt
once, then pass only the newly generated token on each decode step. Comparing a
cached MLX baseline with a course model that recomputes the full prefix would
measure two different algorithms and make the kernel target meaningless. Day 2
implements the dense cache used by every later Week 2 benchmark.

## Synchronize Lazy Work

MLX builds lazy computation graphs. Timing only the Python call measures graph
construction, not GPU execution. Every timed iteration must evaluate the output:

```python
start = perf_counter()
output = function()
mx.eval(output)
elapsed = perf_counter() - start
```

The isolated benchmarks in `benches/` use the same rule. Evaluate input setup
before invoking the benchmark fixture so setup does not leak into the result.

## Record a Matched Baseline

Use the same model, prompt length, output length, device, and warmup count for
the reference and MLX runs:

```bash
pdm run bench --solution tiny_llm_ref --loader week2 --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2

pdm run bench --solution mlx --loader week2 --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

Benchmark on an otherwise idle machine: stop other CPU- and GPU-intensive
workloads, keep power mode and ambient conditions fixed, and let the machine
return to a stable temperature before comparing runs. Run each command several
times, report the median, and include the hardware with the result.

## Acceptance Target

The Week 2 target is:

```plain
reference decode throughput / MLX decode throughput >= 0.80
```

Reaching 80-90% is the acceptance range, not a promise that every educational
kernel individually matches its MLX counterpart. MLX is the comparison
baseline; the Week 2 solution must reach the target with course-owned operator
implementations.

## Expected Performance Contribution

**Estimated decode improvement: 0%.** Benchmarking does not make the model
faster. It prevents us from claiming gains that came from unsynchronized work,
different inputs, or machine noise, and gives every later percentage a common
denominator.

{{#include copyright.md}}
