# 🚧 Measurement: Establish the Comparison

Start with your completed Week 1 model and the environment from
[setup](./setup.md). Before changing its implementation, run a real prompt:

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b
```

You already own the model and generation loop. In this chapter you own the
measurement decision: what request to compare, which output counts as complete,
and what observation would justify the next change. The supplied runners handle
warmups, evaluation, timing, and cache cleanup; there is no new operator to
implement here.

Check that measurement infrastructure first:

```bash
pdm run test --week 2 --day 2
```

This gate exercises the supplied request-timing and release lifecycle with small
fake models. It does not require the Week 2 cache or prove that your Week 1 model
is correct. If it fails, resolve that setup or infrastructure problem before
using its timings.

## Freeze One Request Shape

Use the model already downloaded during setup. These examples select
`qwen3-0.6b`; if you use a different model, change it consistently in every
baseline and candidate. The progression runner's `--offline` option requires
local weights.

```bash
pdm run bench-week2-progression --offline --solution tiny_llm \
  --variant week1 --model qwen3-0.6b \
  --input-len 128 --output-len 129 --warmup 2 --repeats 2 \
  --prefill-logits last --json-output week2-start.json
```

At this point run only `week1`. The cache checkpoint becomes runnable in the
[next chapter](./week2-01-kv-cache.md). Its matched control will be this same
full-prefix Week 1 model, with the same prompt and output lengths.

Prefill processes the prompt and produces the first token. Decode produces the
remaining tokens. Keep both phase results: a change that helps one can hurt the
other. `--prefill-logits last` asks for only the final prompt-position logits;
`all` asks for every prompt position. Compare like with like rather than giving
one implementation the cheaper output contract.

The runner records configuration and repeated samples. Keep model, input/output
lengths, seed, prefill-logit mode, warmups, software, and device fixed when you
compare checkpoints. Use an otherwise idle machine and repeat a comparison
before interpreting a small difference. Two samples are a starting point for
learning the workflow, not strong evidence of a small speedup.

## Time Evaluated Work

MLX can defer computation. A Python function returning an array does not by
itself mean that the GPU has completed the work. The timing boundary must include
evaluation of the result:

```python
start = perf_counter()
output = operation()
mx.eval(output)
elapsed = perf_counter() - start
```

This is a timing sketch, not another runner to implement. Use the supplied tools
so baseline and candidate share their synchronization and cleanup rules. The
cache release hook must run after successful requests and after failures; leaving
resources from the preceding sample changes the next sample's conditions.

## Find Where the Cached Model Spends Time

After completing the cache chapter, return to this command:

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case kv-cache:decode:128 --warmup 4 --iterations 12 \
  --json-output week2-cache-attribution.json
```

A case names `checkpoint:phase:tokens`. Read its recorded workload alongside the
operator-category times. An attribution run helps locate work; its synchronized
category measurements are not interchangeable with complete-request latency.
Changing the prompt length or switching from decode to prefill asks a different
question and needs its own baseline.

If projections dominate, keeping weights packed is a concrete next hypothesis.
If another category dominates, record that observation too; you can still build
the chapter's mechanism without claiming it was your largest bottleneck. After
each implementation, repeat both the affected component comparison and the
complete request. A lower percentage can merely mean another category got
slower, so inspect absolute times as well as shares.

Finish this chapter with your saved Week 1 result and a short comparison contract:
“On this model and request shape, I will compare full-prefix generation with
cached generation. I expect less repeated model work; I will revisit that
hypothesis if the matched complete-request timing does not improve.” Then build
the [capacity-backed KV cache](./week2-01-kv-cache.md). The first Week 2 checkpoint
is `kv-cache`; no custom Metal primitive is needed to reach it.

{{#include copyright.md}}
