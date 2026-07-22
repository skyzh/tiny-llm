# Week 2 Day 1: Benchmarking and the MLX Baseline (WIP)

> This newly introduced chapter is a work in progress.

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

Run both commands more than once when the machine is changing temperature or
running other GPU workloads. Keep the median or a representative stable run,
and report the hardware with the result.

## Acceptance Target

The Week 2 target is:

```plain
reference decode throughput / MLX decode throughput >= 0.80
```

Reaching 80-90% is the acceptance range, not a promise that every educational
kernel individually matches its production counterpart. We will use native MLX
primitives at the required boundary when a teaching kernel misses the target,
and keep the teaching kernel as an explicit stretch exercise.

{{#include copyright.md}}
