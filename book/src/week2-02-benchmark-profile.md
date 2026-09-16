# 🚧 Week 2 Day 2: Benchmarking and Profiling

Day 1 leaves you with a cached BF16 model and a working `kv-cache` checkpoint.
Day 2 does not add another model operator. The supplied benchmark and portable
attribution runners own request generation, warmups, synchronization, phase
timing, and cache release. Your job is to freeze one like-for-like workload,
identify its dominant operator category, and write the short decision that
chooses the next change.

Start with the focused benchmark-lifecycle check:

```bash
pdm run test --week 2 --day 2
```

When it passes, record one matched `tiny_llm`/MLX pair and one attribution
result, then write the short decision that follows from them. Those portable
JSON records are the Day 2 checkpoint. Metal capture remains optional and
never gates the next chapter.

## Benchmark the Cached Model

Before changing the model, make the comparison trustworthy. Prefill processes
many prompt tokens at once, while decode usually processes one token per
request. At this checkpoint, decode repeatedly reads dense BF16 projection
weights. Because a change can help one phase while hurting the other,
`benches/bench.py` reports them separately:

- prefill tokens per second: prompt tokens divided by prefill time;
- decode tokens per second: generated tokens after the first token divided by
  decode time.

The first generated token is part of prefill. Leaving it out of decode keeps
prompt length from distorting the decode number.

Decide what prefill should return before comparing implementations. Prompt
scoring needs logits for every position; serving needs only the final prompt
logit. Use `--prefill-logits all` for the former and
`--prefill-logits last` for the latter. The runner applies one choice to your
solution and MLX alike, so the two rows do the same work.

Keep the Week 2 generation algorithm matched too. Both sides use a KV cache:
prefill the prompt once, then pass only the newly generated token on each
decode step. A cached MLX baseline against a full-prefix solution would compare
two different algorithms instead of locating the next optimization target.

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
pdm run bench-week2-progression --offline --repeats 2 \
  --solution tiny_llm \
  --variant week2-kv-cache --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-baseline.json
```

Benchmark on an otherwise idle machine. Stop other CPU- and GPU-intensive
workloads, keep power mode and ambient conditions fixed, and wait for a stable
temperature before comparing runs. Repeat each command, report the median, and
record the hardware, MLX and mlx-lm versions, prefill-logit mode, and exact
model. After a dependency upgrade, remeasure MLX instead of carrying the old
baseline forward.

### Synchronize Lazy Work

MLX builds computation graphs lazily. Timing only the Python call measures
graph construction instead of GPU execution, so every timed iteration must
evaluate its output:

```python
start = perf_counter()
output = function()
mx.eval(output)
elapsed = perf_counter() - start
```

The benchmark must also call the cache release hook after warmups and timed
runs. That lets caches return owned or shared resources even when a run fails;
the focused Day 2 test covers both paths.

## Attribute the Cached Model

Next, attribute the same cached-decode workload. Keep the learner solution,
model, decode phase, and 128-token context fixed:

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case kv-cache:decode:128 --warmup 4 --iterations 12 \
  --json-output week2-day2-attribution.json
```

The result identifies its source, checkpoint, phase, token count, prompt rule,
software, host, category medians, and category shares without depending on a
private function name or Metal symbol. On the checked M4 Pro run, dense
projections accounted for 81.5% of attributed cached-decode time. That bounded
observation selected packed W4 projections for Day 3; another device or shape
may point somewhere else.

Turn the observation into a decision with three sentences:

1. “Dense projections dominate this exact cached-decode workload.”
2. “Packing W4 weights and changing only the selected projection path should
   reduce that category and improve matched decode.”
3. “I will reject or revise the hypothesis if projection time does not fall or
   complete-model decode regresses under the same workload.”

Substitute the category you observed for the checked example. Your required
work ends with the benchmark, attribution, and decision record. The
[macOS 27 capture lab](./week2-advanced-profiling.md) is optional; no trace,
`gpudebug` output, screenshot, or device-specific counter gates Day 3.

## Why Quantize: The Decode Roofline

The measurement now has a hardware reason to test. LLM decode is typically
**memory-bandwidth bound**: each token reads the model's weights while doing
relatively little work with them. Use the dimensions in the official
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

Count the tied embedding matrix once as the vocabulary projection. The
single-row embedding lookup, normalization weights, activations, KV reads, and
attention work are omitted, so the result is an upper bound for linear layers
rather than a prediction of complete-model throughput. A dense FP16 or BF16
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
