# 🚧 Week 2 Day 1: Reuse the Prefix, Then Bound the Cache

Your Week 1 Qwen model already generates by rerunning the full prefix. Day 1
keeps that path intact while you first complete the `kv-cache` checkpoint, then
bound the request-owned storage for `capacity-cache`. The four initial shells are:

- `src/tiny_llm/kv_cache.py::TinyKvFullCache` stores one layer's dense K/V;
- `src/tiny_llm/qwen3_week2.py::Qwen3ModelWeek2` threads cache state and
  offsets through the model;
- `Qwen3ModelWeek2.create_kv_cache` creates one cache per layer and request;
- `src/tiny_llm/generate.py` prefills once, then sends only the new token.

Together, these pieces make prefill populate the cache and make decode send
only the new token. The starter already supplies the Week 1 operators and the
model-loading boundary. Start with the focused learner gate:

```bash
pdm run test --week 2 --day 1
```

When it passes, run the `kv-cache` checkpoint shown in Task 4. That live call
puts the cache into the generation loop instead of exercising it only as an
isolated data structure.

Each attention layer can then reuse the keys and values from previous tokens
instead of recomputing the entire prefix at every step.

This is the foundation of Week 2 decode optimization. Week 3 will change how
the cache is stored and shared, but the reuse starts here. Without it, every
generated token reruns all model layers over an ever-growing prefix and can
overwhelm gains from faster individual kernels.

**📚 Readings**

- [KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching)

First, make the repeated work concrete. Week 1 supplied the full sequence to
the model on every step:

```plain
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6]) # returns 7
decode:  _step(model, [1, 2, 3, 4, 5, 6, 7]) # returns 8
decode:  _step(model, [1, 2, 3, 4, 5, 6, 7, 8]) # returns 9
...
```

```plain
x: B, L, E
q = linear(x, wq) -> B, L, H_q, D
k = linear(x, wk) -> B, L, H, D
v = linear(x, wv) -> B, L, H, D
q = rms_norm(q, q_norm)
k = rms_norm(k, k_norm)
q = rope(q, offset=slice(offset, offset + L))
k = rope(k, offset=slice(offset, offset + L))
(transpose as needed)
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D
# q/k/v and the returned model tensor are BF16; the Python `mlx.core` expression may use FP32 intermediates
(transpose as needed)
x = linear(x, wo) -> B, L, E
```

The attention mechanism is computed as:

$$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$


Consider two consecutive decoding steps with `L = S = 3` and `L = S = 4`.
Assume that each attention head has dimension `D = 4`:

```
L = 3
Q        x  K^T     =
1 1 1 1     1 2 3      1x1  -inf -inf
2 2 2 2     1 2 3      2x1  2x2  -inf
3 3 3 3     1 2 3      3x1  3x2  3x3
            1 2 3

L = 4
Q        x  K^T       =
1 1 1 1     1 2 3 4      1x1  -inf -inf -inf
2 2 2 2     1 2 3 4      2x1  2x2  -inf -inf
3 3 3 3     1 2 3 4      3x1  3x2  3x3  -inf
4 4 4 4     1 2 3 4      4x1  4x2  4x3  4x4
```

The leading `3 x 3` block of `QK^T` is identical in both steps. The causal mask
prevents earlier queries from attending to the new token, so those outputs do
not change either. Only the new query row can produce a new output; recomputing
the earlier rows, softmax values, and products with `V` is wasted work.

Instead, cache the previous keys and values and compute only the projections for
incoming tokens:

```
K in cache:
1 1 1 1
2 2 2 2

[a b c d] represent cached values

L = 1, S = 3
Q        x  K^T       =
            (⬇️ is K not transposed)
            [1 1 1 1]
            [2 2 2 2]
3 3 3 3      3 3 3 3      3x1 3x2 3x3

L = 1, S = 4
Q        x  K^T       =
            (⬇️ is K not transposed)
            [1 1 1 1]
            [2 2 2 2]
            [3 3 3 3]
4 4 4 4      4 4 4 4      4x1 4x2 4x3 4x4
```

## Task 1: Implement the Key-Value Cache

```
src/tiny_llm/kv_cache.py
```

Each Transformer layer owns a key-value cache. Its `update_and_fetch` method:

1. Accepts the newly computed `K` and `V` for the incoming tokens.
2. Appends them along the sequence dimension.
3. Returns the complete cached `K` and `V`, the updated offset, and the mask.

For now, pass `mask` through unchanged and leave `mask_length` unused. Week 3
will use both when requests share a batch.

You may implement this in `kv_cache.py` as `TinyKvFullCache`:

```plain
L_new = number of incoming tokens

update_and_fetch(key, value, mask_length, mask) -> key, value, offset, mask

key:   B, H, L_new, D
value: B, H, L_new, D

if self.key_values is None:
    self.key_values = (key, value)
else:
    cached_key, cached_value = self.key_values
    self.key_values = (
        concat(cached_key, key, axis=2),
        concat(cached_value, value, axis=2),
    )

self.offset += L_new
key, value = self.key_values  # B, H, offset, D

return key, value, self.offset, mask
```

Keep this first cache deliberately simple and dense. Each `mx.concat` allocates
a larger buffer and copies the previous K/V contents. Across a token-by-token
decode of length `S`, those copies add up to `O(S²)` bytes even though the cache
avoids `O(S²)` prefix recomputation. The reference cache records that traffic
as `growth_copy_bytes` so the profiler can separate it from attention. The second checkpoint in this lesson replaces repeated concatenation with a
request-bounded allocation. Week 3 later introduces pages for serving.

## Task 2: Build the Cached Week 2 Model

```
src/tiny_llm/qwen3_week2.py
```

Keep the Week 1 Python model and its full-prefix generation loop unchanged.
Build the separate `qwen3_week2.py` model with the same dense weights and the
Week 1 `mlx.core` RMSNorm, RoPE, SwiGLU, and attention equations. Change only
the state flow: the Week 2 model accepts a cache and an offset, while Week 1
continues to recompute the full prefix. Every later Week 2 chapter starts from
this baseline.

- Give each layer its own cache.
- Add an `offset` argument to the model. It is the number of tokens already in
  the cache, and therefore the position of the first incoming token.
- The argument should match the cache's current sequence length. Assertions can
  make this invariant explicit.
- The caller and cache both track the offset to make consistency checks easier.

Example computation flow:

```plain
x: B, L, E
q = linear(x, wq) -> B, L, H_q, D
k = linear(x, wk) -> B, L, H, D
v = linear(x, wv) -> B, L, H, D
q = rms_norm(q, q_norm)
k = rms_norm(k, k_norm)
q = rope(q, offset=slice(offset, offset + L))
k = rope(k, offset=slice(offset, offset + L))
transpose q, k, v to B, H, L, D
k, v = cache.update_and_fetch(k, v)  # k/v: B, H, S, D; q: B, H_q, L, D
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, H_q, L, D
# q/k/v and the returned model tensor are BF16; attention arithmetic is still the Week 1 `mlx.core` path
transpose and reshape x to B, L, H_q * D
x = linear(x, wo) -> B, L, E
```

Here, `L` is the number of incoming query tokens and `S` is the total cached
sequence length after the update. This matches the Week 1 GQA convention: `L`
is the query length, while `S` is the key/value source length. During
single-token decoding, `L = 1` and `S` grows by one on each call.

The linear layers, RMSNorm, RoPE, SwiGLU, and attention remain the Week 1
Python implementations at this checkpoint. Save packed weights and fast
kernels for later checkpoints so this measurement isolates one algorithmic
change. The model still uses BF16 storage; "Week 1 Python" describes the
implementation style, not a return to an FP32 model.

## Task 3: Create Request-Scoped Caches

```
src/tiny_llm/qwen3_week2.py
```

Implement `create_kv_cache` so each request receives one cache handle per
Transformer layer. Pass the matching cache through each block, and keep the
caller's offset equal to the cache's logical length.

The first Day 1 gate checks this request-scoped lifecycle together with the
cache and model work from the earlier tasks.

## Task 4: Connect the Serving Loop

```
src/tiny_llm/generate.py
```

Send the complete prompt on the first model call to prefill the cache. On each
later call, send only the token produced by the preceding step and the number
of tokens already cached. Week 3 moves this same lifecycle into the
continuous-batching scheduler.

For example:

```plain
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6], 0)  # returns 7
decode:  _step(model, [7], 6)  # returns 8
decode:  _step(model, [8], 7)  # returns 9
...
```

You can test your solution with:

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b --max-tokens 16
```

You can also run the same loop with the reference solution:

```bash
pdm run main --solution tiny_llm_ref --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b --max-tokens 16
```

## Measure the First Cache Change

Before adding capacity, compare the Week 1 full-prefix loop with this
`kv-cache` checkpoint. The progression runner starts each variant in a fresh
process, uses the same Qwen3-4B 128-token prompt and 129-token output, and
records the workload and results in JSON:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 \
  --repeats 2 --variant week1 --variant week2-kv-cache \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits all --json-output week2-day1-cache.json
```

Keep `week2-day1-cache.json` as the baseline for the capacity checkpoint
below. Week 1 requires `--prefill-logits all`; this is a matched algorithm
comparison, not a serving-only prefill measurement. Record the observation
and workload identity without assuming its speedup holds on another machine.


## Second checkpoint: bound the request cache

The dense cache has removed repeated model computation, but its concatenation
still copies the old K/V prefix on each append. Physical capacity and logical
length are different: only the logical prefix is visible to attention. The
`capacity-cache` checkpoint keeps the same serving loop and readable operators.
It allocates storage from the request's prompt-plus-output bound, writes new
K/V into a slice, and returns a view of the active prefix. Capacity can raise
peak memory for some shapes while removing repeated prefix copies.

## First Diagnostic: Hide Unused Capacity

```bash
pdm run test --week 2 --day 2 -- -k logical_prefix
```

The expected first failure points at `_logical_key_values` or the capacity
branch of `update_and_fetch`. For physical arrays shaped `B, H, capacity, D`,
attention must see only `:offset`:

```text
physical storage: [token 0][token 1][unused][unused]
logical prefix:   [token 0][token 1]
offset = 2, capacity = 4
```

Do not infer logical length from the backing array's shape.

## Allocate from the Request Bound

The generation loop knows the prompt length and maximum number of new tokens.
Use that request-local bound when it creates each layer cache. Capacity is not a
global maximum and must not grow beyond the request's declared budget.

On the first append, allocate K/V storage for the full capacity. On every
append:

1. compute `end = offset + L_new`;
2. reject `end > capacity` before changing storage, offset, or counters;
3. use `mx.slice_update` on sequence axis 2;
4. advance `offset` only after the write is valid;
5. return `storage[:, :, :offset, :]` for both K and V.

That ordering makes overflow transactional. A rejected append leaves the old
logical cache usable.

## Make Movement Observable

The cache exposes three counter categories:

| Counter | Meaning | Expected capacity behavior |
|---|---|---|
| `logical_copy_bytes` | old logical K/V copied by concatenation | zero |
| `physical_growth_copy_bytes` | old K/V copied while growing storage | zero |
| `slice_write_bytes` | newly written K/V bytes | increases by each append's K/V size |
| `growth_copy_bytes` | legacy total for growth copies | zero |

Counters are mechanism evidence. They explain which bytes moved; they do not by
themselves establish lower complete-request latency or peak memory.

## Reset and Rewind Without Leaking a Suffix

`rewind(n)` shortens the logical length and rejects negative or oversized
rewinds. The next append may reuse the abandoned physical slots, but attention
must not see values beyond the new offset. `reset()` returns the logical cache
to length zero. It clears storage for the unbounded fallback but retains the
request-bounded allocation. All four movement counters remain
lifetime-cumulative across rewind and reset, so measure their deltas when you
need per-request evidence.

Run the state-transition witness before the product:

```bash
pdm run test --week 2 --day 2 -- -k 'rewind or overflow'
```

Test the sequence append → rewind → append as well as an overflow after valid
data. Those cases catch implementations that expose physical capacity as
logical state or mutate before validation.

## Complete the `capacity-cache` Checkpoint

```bash
pdm run test --week 2 --day 2
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint capacity-cache --model qwen3-0.6b --max-tokens 16
```

The predecessor fallback is `kv-cache`: it keeps the same generation algorithm
and readable model but uses concatenation. If bounded allocation cannot be
established, return to that checkpoint rather than exposing unused storage.


## Compare the two cache checkpoints

After the focused gates, use identical prompts and output bounds for the two
public checkpoints. These coarse commands include process startup; use the
supplied progression runner below when you need separated prefill/decode
measurements and fresh-process repeats.

```bash
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b --max-tokens 16
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint capacity-cache --model qwen3-0.6b --max-tokens 16
```

The cache counters distinguish old logical-prefix copies, physical growth
copies, and new slice writes. A counter difference establishes the mechanism;
it does not alone prove a complete-request speedup. A historical exact-mechanism
run observed a **+88.0 MiB / +2.276%** temporal 2K/512 peak-memory tradeoff.
That is historical evidence, not a result from this checkout.

Extend the first JSON baseline with the new capacity checkpoint using the
same model, lengths, warmups, and `all`-logit workload. Save a second JSON
file so the original Week 1 versus `kv-cache` observation remains available:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 \
  --repeats 2 --variant week1 --variant week2-kv-cache \
  --variant week2-capacity-cache --model qwen3-4b \
  --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits all --json-output week2-day1-cache-ladder.json
```

Compare these rows with `week2-day1-cache.json` only when the recorded
workload and device match. The serving comparisons below use
`--prefill-logits last`, so keep their results separate from this ladder.

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
  --week2-checkpoint capacity-cache --model qwen3-4b \
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
  --solution tiny_llm --suite week2 \
  --variant week2-capacity-cache --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day1-baseline.json
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
the supplied benchmark-lifecycle test covers both paths.

## Attribute the Cached Model

Next, attribute the same cached-decode workload. Keep the learner solution,
model, decode phase, and 128-token context fixed:

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case capacity-cache:decode:128 --warmup 4 --iterations 12 \
  --json-output week2-day1-attribution.json
```

The result identifies its source, checkpoint, phase, token count, prompt rule,
software, host, category medians, and category shares without depending on a
private function name or Metal symbol. An earlier checked M4 Pro run attributed
81.5% of cached-decode time to dense projections at the then-current `kv-cache`
checkpoint. This is a
historical example, not a measurement of your bounded-cache checkout; another
device or shape may point elsewhere.

Turn the observation into a decision with three sentences:

1. “Dense projections dominate this exact cached-decode workload.”
2. “Packing W4 weights and changing only the selected projection path should
   reduce that category and improve matched decode.”
3. “I will reject or revise the hypothesis if projection time does not fall or
   complete-model decode regresses under the same workload.”

Substitute the category you observed for the checked example. Your required
work ends with the benchmark, attribution, and decision record. The
[macOS 27 capture lab](./week2-advanced-profiling.md) is optional; no trace,
`gpudebug` output, screenshot, or device-specific counter gates the W4 lesson.

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
then continue to [Day 2](./week2-02-quantize-model.md), where the model keeps
weights packed, replaces the live projection path, and reruns the same
benchmark.

{{#include copyright.md}}
