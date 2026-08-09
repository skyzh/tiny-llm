# 🚧 Week 2 Day 2: Benchmarking and Profiling

> **Status: Experimental.** See the
> [Week 2 verification matrix](./week2-overview.md#verification-status) for
> what is continuously tested, locally measured, and still under review.

Day 1 gave us a cached model. Day 2 establishes a trustworthy dense BF16
baseline: how fast are prefill and decode under one matched protocol, and what
architectural cost should the next chapter attack? Benchmarking is required.
Profiling is optional and is not a prerequisite or acceptance gate.

## Benchmark the Cached Model

Optimization starts with a trustworthy comparison. Prefill processes many
prompt tokens at once; decode usually processes one token per request and is
dominated by repeatedly reading dense BF16 projection weights at this
checkpoint. A change can improve one phase while hurting the other, so
`benches/bench.py` reports both:

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
different algorithms and make the next optimization target meaningless.

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
construction, not GPU execution. Every timed iteration must evaluate the
output:

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

## Optional Profiling Boundary

The required Day 2 work ends with the synchronized benchmark JSON. Metal
capture, Xcode visualization, `gpudebug`, and related profiling
microbenchmarks are not part of the current course requirements. They require
the macOS 27 tooling release and will return as optional material after that
release is available.

The [optional profiling notice](./week2-advanced-profiling.md) records this
boundary. You may skip it and continue directly to Day 3. No profiling tool,
trace, screenshot, or microbenchmark is a prerequisite or acceptance gate.

## Why Quantize: The Decode Roofline

The decode phase of LLM inference is typically **memory-bandwidth bound**: each
token requires reading the model's weights but performs relatively little work
with them. Use the dimensions in the official
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

This is the baseline to improve: both dense formats must stream roughly 8 GB
of projection weights to generate one token. Save the matched benchmark result,
then continue to [Day 3](./week2-03-quantize-model.md), where the model keeps
weights packed, replaces the live projection path, and reruns the same
benchmark.

{{#include copyright.md}}
