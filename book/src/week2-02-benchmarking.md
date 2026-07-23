# 🚧 Week 2 Day 2: Benchmark Decode

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

For a matched prefill comparison, all implementations compute logits for every
prompt position. The generation path may request only the final logit row, but
using that shortcut for the course model while MLX projects all rows would make
the prefill columns incomparable. Cached decode has `L = 1`, so
`logits_to_keep=1` removes no decode work.

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

The isolated benchmarks in `benches/` use the same rule. Evaluate input setup
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

## Record a Matched Baseline

Use the same model, prompt length, output length, device, and warmup count for
your implementation and the MLX run. Replace `tiny_llm` with `tiny_llm_ref` to
compare against the course reference:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2

pdm run bench --solution mlx --loader week2 --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

Or run the complete cumulative ladder in fresh processes. At this point, only
the Week 1, KV-cache, and MLX rows are course prerequisites; later rows become
meaningful as you complete their chapters.

```bash
pdm run bench-week2-progression --offline --repeats 3 \
  --model qwen3-0.6b --input-len 128 --output-len 65 --warmup 2
```

Benchmark on an otherwise idle machine: stop other CPU- and GPU-intensive
workloads, keep power mode and ambient conditions fixed, and let the machine
return to a stable temperature before comparing runs. Run each command several
times, report the median, and include the hardware with the result.

## Acceptance Target

The Week 2 target is:

```plain
reference decode throughput / MLX decode throughput >= 0.70
```

Reaching 70% is the acceptance threshold, not a promise that every educational
kernel individually matches its MLX counterpart. MLX is the comparison
baseline; the Week 2 solution must reach the target with course-owned operator
implementations.

{{#include copyright.md}}
