# 🚧 Week 2 Day 1: Reuse the Prefix, Then Bound the Cache

Watch how your Week 1 model generates text. Every step feeds it the whole
sequence so far:

```plain
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6])        # returns 7
decode:  _step(model, [1, 2, 3, 4, 5, 6, 7])     # returns 8
decode:  _step(model, [1, 2, 3, 4, 5, 6, 7, 8])  # returns 9
...
```

To produce token 9, the model runs all its layers over eight tokens, even
though it ran them over the first seven a moment ago. Each step costs more
than the one before, because the sequence it reprocesses keeps growing. No
kernel you write later can make up for work that grows like that, so Week 2
starts by removing it.

Today you build two checkpoints:

1. **`kv-cache`** keeps each layer's keys and values from earlier steps, so a
   decode step sends only the newest token through the model.
2. **`capacity-cache`** stores that cache in one buffer sized for the whole
   request, so your code no longer concatenates the old cache with each new
   token.

You will write four pieces:

- `TinyKvFullCache` in `src/tiny_llm/kv_cache.py`, which stores one layer's
  K/V;
- `Qwen3ModelWeek2` in `src/tiny_llm/qwen3_week2.py`, which passes the cache
  and the current position through the model;
- `Qwen3ModelWeek2.create_kv_cache`, which gives each request one cache per
  layer; and
- `simple_generate_with_kv_cache` in `src/tiny_llm/generate.py`, which
  prefills once and then sends one token at a time.

The starter supplies the Week 1 operators and the model loader. Finish the
[Week 1 Day 7](./week1-07-sampling-prepare.md) setup first. Today's work is
all Python, but the Day 1 tests import the native Week 2 extension, so build
it before the first test:

```bash
pdm run build-ext
pdm run test --week 2 --day 1 -- -k 'not capacity'
```

Those three tests cover the `kv-cache` checkpoint. They fail until you finish
Tasks 1 to 3; rerun them after Task 4. The full day's tests come after the
`capacity-cache` work.

**📚 Readings**

- [KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching)

## Why the Old Work Can Be Skipped

Here is one attention layer from Week 1, with shapes:

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
(transpose as needed)
x = linear(x, wo) -> B, L, E
```

$$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Compare two decode steps, one with three tokens and one with four, using a
head dimension of 4. The causal mask `M` puts `-inf` above the diagonal:

```plain
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

The top-left 3 × 3 block is the same in both steps. Because of the mask, the
first three rows never see the new token, so their outputs do not change. Only
the last row is new. Everything else the second step computes is a repeat.

The same holds in every layer. Because of the causal mask, a token's keys
and values depend only on that token and the ones before it, so they never
change when later tokens arrive. Keep them from earlier steps, and compute `q`,
`k`, and `v` for the new token only:

```plain
cached K:
1 1 1 1
2 2 2 2
3 3 3 3

new token: L = 1, S = 4
Q          K^T (cached rows plus the new one)
4 4 4 4    [1 1 1 1]
           [2 2 2 2]
           [3 3 3 3]
           [4 4 4 4]   ->   4x1 4x2 4x3 4x4
```

Here `L` is the number of incoming query tokens and `S` is the number of
cached tokens after adding them. During decode, `L = 1` and `S` grows by one
each step.

## Task 1: Implement the Key-Value Cache

```
src/tiny_llm/kv_cache.py
```

Each transformer layer gets its own cache. Its `update_and_fetch` method:

1. receives `K` and `V` for the incoming tokens, each shaped `B, H, L_new, D`;
2. appends them along the sequence axis (axis 2);
3. returns the full cached `K` and `V`, the new offset, and the mask.

Pass `mask` through unchanged and ignore `mask_length` for now; Week 3 uses
both when several requests share a batch.

Start with the simplest version in `TinyKvFullCache`, used when `capacity` is
`None`:

```plain
update_and_fetch(key, value, mask_length, mask) -> key, value, offset, mask

if self.key_values is None:
    self.key_values = (key, value)
else:
    cached_key, cached_value = self.key_values
    self.key_values = (
        concat(cached_key, key, axis=2),
        concat(cached_value, value, axis=2),
    )

self.offset += L_new
key, value = self.key_values   # B, H, offset, D
return key, value, self.offset, mask
```

This works, but each `mx.concat` allocates a bigger buffer and copies
everything already cached into it. Over `S` decode steps those copies add up to
`O(S²)` bytes. That is still far cheaper than recomputing the model over the
prefix, but it is not free. The starter's constructor already creates byte
counters for this; add the size of the copied old K/V to
`logical_copy_bytes`, `physical_growth_copy_bytes`, and `growth_copy_bytes`
on each concatenation, so the benchmark can report it. The `capacity-cache`
checkpoint later today removes the concatenation.

## Task 2: Build the Cached Week 2 Model

```
src/tiny_llm/qwen3_week2.py
```

Leave the Week 1 model and its generation loop alone. Build the Week 2 model in
`qwen3_week2.py` from the same BF16 weights and the same Week 1 `mlx.core`
operators: linear, RMSNorm, RoPE, SwiGLU, and attention. The only change is how
state flows through it:

- The model's call takes the per-layer caches and an `offset`: the number of
  tokens already cached, which is also the position of the first incoming
  token. RoPE uses it to rotate the new tokens to the right positions.
- Each attention layer appends its new `k` and `v` to its cache and attends
  over everything the cache returns:

```plain
transpose q, k, v to B, H, L, D
k, v, _, mask = cache.update_and_fetch(k, v, mask=mask)  # k, v: B, H, S, D
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask)  # B, H_q, L, D
transpose and reshape x to B, L, H_q * D
x = linear(x, wo) -> B, L, E
```

- If the caller's `offset` does not match the cache's length, raise
  `ValueError`. A mismatched offset puts RoPE positions and cached keys out of
  step, and the model would quietly produce wrong text.
- Support `logits_to_keep`: when it is set, apply the final norm and output
  head only to the last `logits_to_keep` positions. Generation needs only the
  last row.

Keep every operator exactly as in Week 1 for now. Packed weights and faster
kernels come on later days, and keeping them out today means the measurement
below shows the effect of caching alone. "Week 1 operators" describes the code,
not the precision: the model still stores BF16 weights and activations.

Every later Week 2 day builds on this model. It selects features through its
`checkpoint` argument (see `WEEK2_CHECKPOINT_FEATURES` in the same file); today
you need `kv-cache` and, later, `capacity-cache`.

## Task 3: Create Request-Scoped Caches

```
src/tiny_llm/qwen3_week2.py
```

Implement `create_kv_cache(capacity=None)` so each request gets a fresh list
of caches, one `TinyKvFullCache(capacity=capacity)` per transformer layer.
Pass layer `i`'s cache to layer `i`. Two requests must never share cache
objects.

```bash
pdm run test --week 2 --day 1 -- -k 'not capacity'
```

These tests append chunks to a cache, check that prefill-then-decode gives the
same last-row logits as running the full sequence at once, and check that a
mismatched offset raises `ValueError`.

## Task 4: Connect the Serving Loop

```
src/tiny_llm/generate.py
```

Implement `simple_generate_with_kv_cache`. Create the caches once, send the
whole prompt on the first call, then send only the token the previous step
produced, along with the number of tokens already cached:

```plain
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6], 0)  # returns 7
decode:  _step(model, [7], 6)                 # returns 8
decode:  _step(model, [8], 7)                 # returns 9
...
```

When generation ends, normally or with an error, call the supplied
`_release_kv_cache` on the caches. Week 3 moves this same lifecycle into its
batching scheduler.

Run your model, and the reference solution for comparison:

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b --max-tokens 16

pdm run main --solution tiny_llm_ref --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b --max-tokens 16
```

## Measure the First Cache Change

Now time Week 1 against `kv-cache`. The progression runner starts each variant
in a fresh process, runs both with the same cached Qwen3-0.6B model, a
128-token prompt, and 129 output tokens, and writes the workload and results to
JSON:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 \
  --repeats 2 --variant week1 --variant week2-kv-cache \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits all --json-output week2-day1-cache.json
```

Week 1 always computes logits for every prompt position, so this comparison
uses `--prefill-logits all` for both rows. That makes it a fair comparison of
the two algorithms, but not a measurement of a serving setup, which needs only
the last logit. Keep `week2-day1-cache.json`; the capacity checkpoint below
adds to it. The speedup you see depends on your machine, so record it rather
than expecting a particular number.

## Second Checkpoint: Bound the Request Cache

`kv-cache` removed the repeated model work, but every append still asks for a
bigger buffer and concatenates the old K/V into it. The alternative is to
size the storage once.

The generation loop knows how long a request can get: the prompt length plus
`max_tokens`. If each layer's cache allocates that much space up front, an
append only has to describe a write of the new tokens into the next free
slots. Whether the GPU then updates that buffer in place or copies it first is
up to MLX: `mx.slice_update` reuses the existing buffer when MLX can donate it
to the result, and copies it otherwise. Your code no longer asks for the
copy, but nothing in this chapter measures which case happens.

With storage sized up front, two numbers matter, and they are different:

```text
physical storage: [token 0][token 1][unused][unused]
logical prefix:   [token 0][token 1]
offset = 2, capacity = 4
```

The **capacity** is how many slots exist. The **offset** is how many hold real
tokens. Attention must see only the first `offset` slots. If it sees the unused
tail, it attends to zeros that are not part of the sequence.

Allocating the full request up front can raise peak memory for some shapes,
in exchange for removing the repeated concatenation. The `capacity-cache` checkpoint
keeps the same serving loop and the same Week 1 operators; only the cache's
storage changes.

<a id="first-diagnostic-hide-unused-capacity"></a>

### Show Only the Logical Prefix

Start with the failing test:

```bash
pdm run test --week 2 --day 1 -- -k logical_prefix
```

It points at `_logical_key_values` or the capacity branch of
`update_and_fetch`. For storage shaped `B, H, capacity, D`,
`_logical_key_values` returns `storage[:, :, :offset, :]` for both keys and
values. Never work out the logical length from the storage's shape.

<a id="allocate-from-the-request-bound"></a>

### Allocate from the Request's Limit

When `capacity` is set, on every append:

1. compute `end = offset + L_new`;
2. if `end > capacity`, raise `ValueError` with a message that includes
   `capacity N exceeded` (N is the capacity), before changing anything;
3. on the first append, allocate zeroed K and V storage with room for
   `capacity` tokens;
4. write the new K and V into slots `offset` to `end` with `mx.slice_update`
   on axis 2, and add their size to `slice_write_bytes`;
5. set `offset = end`;
6. return the logical prefix.

Because the check comes first, a rejected append leaves the cache exactly as
it was, and the caller can still use it.

Then wire the limit through. The model constructor has a
`use_bounded_kv_capacity` argument; when it is `None`, take the value from the
checkpoint's `bounded_kv_capacity` feature (true for `capacity-cache` and every
later checkpoint), and store it on the model as `use_bounded_kv_capacity`. In
`simple_generate_with_kv_cache`, use the caller's `use_bounded_kv_capacity`
if given, otherwise the model's, and when it is true create the caches with
`capacity = len(prompt tokens) + max_tokens`. The capacity belongs to this
request only, and the cache must never grow past it.

<a id="make-movement-observable"></a>

### Count What Moves

The cache keeps four byte counters:

| Counter | What it counts | With `capacity-cache` |
|---|---|---|
| `logical_copy_bytes` | Old K/V copied by concatenation | stays zero |
| `physical_growth_copy_bytes` | Old K/V copied while storage grows | stays zero |
| `slice_write_bytes` | New K/V written | grows by each append's K/V size |
| `growth_copy_bytes` | Older name for the growth-copy total | stays zero |

These are bookkeeping counts kept by your Python code: what it asked to
concatenate and what it asked to write. They do not observe what the GPU
actually copies, so zero growth bytes means your code stopped concatenating,
not that no copy happened underneath. They also do not tell you whether a
whole request got faster or used less memory; only a timed, matched
comparison does.

<a id="reset-and-rewind-without-leaking-a-suffix"></a>

### Rewind and Reset

`rewind(n)` drops the newest `n` tokens by lowering `offset`. Raise
`ValueError` if `n` is negative, larger than `offset`, or not an `int`. The
next append may overwrite the abandoned slots, but attention must never see
slots past the new offset. Speculative decoding in Week 3 uses this to throw
away rejected draft tokens.

`reset()` sets the logical length back to zero. With a capacity, keep the
allocated storage for reuse; without one, drop it. The counters are totals
over the cache's lifetime and are not cleared by rewind or reset, so subtract
before-and-after values when you want numbers for one request.

```bash
pdm run test --week 2 --day 1 -- -k 'rewind or overflow'
```

This test appends, rewinds, appends again, then overflows on purpose,
checking each time that only the right tokens are visible. It catches caches
that show unused slots or change state before rejecting an append.

<a id="complete-the-capacity-cache-checkpoint"></a>

### Finish `capacity-cache`

```bash
pdm run test --week 2 --day 1
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint capacity-cache --model qwen3-0.6b --max-tokens 16
```

If you cannot get bounded storage working, `kv-cache` is still a correct
fallback: same algorithm, same model, just with concatenation. Fall back to it
rather than let attention see unused storage.

## Compare the Two Cache Checkpoints

For a quick look, run both checkpoints with the same prompt and output limit:

```bash
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b --max-tokens 16
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint capacity-cache --model qwen3-0.6b --max-tokens 16
```

These times include process startup, so they are rough. For separate prefill
and decode numbers over repeated fresh runs, add the capacity checkpoint to
the earlier comparison, using the same model, lengths, warmup, and logit mode,
and save it to a new file so the first result stays intact:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --suite week2 \
  --repeats 2 --variant week1 --variant week2-kv-cache \
  --variant week2-capacity-cache --model qwen3-0.6b \
  --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits all --json-output week2-day1-cache-ladder.json
```

Compare it with `week2-day1-cache.json` only if the recorded workload and
device match. One older run of this change, with earlier course code, saw
peak memory rise by 88.0 MiB (2.276%) at a 2,048-token prompt with 512 output
tokens. That is the trade-off described above, shown on one machine, not a
result for your checkout.

## Benchmark the Cached Model

From now on, every day ends by timing a new checkpoint against the previous
one. It is worth getting the method right once.

**Report prefill and decode separately.** Prefill processes the whole prompt
at once; decode handles one token at a time and, at this checkpoint, rereads
all the dense BF16 projection weights each step. A change can help one and
hurt the other, so `benches/bench.py` reports:

- prefill tokens per second: prompt tokens divided by prefill time;
- decode tokens per second: tokens generated after the first one, divided by
  decode time.

The first generated token comes out of prefill, so it is not counted as
decode. Otherwise a long prompt would inflate the decode number.

**Decide which logits prefill returns.** Scoring a prompt needs logits for
every position. Serving needs only the last one. Use `--prefill-logits all`
for the first and `--prefill-logits last` for the second. The runner applies
the same choice to every solution in a comparison, so they do the same work.

**Compare like with like.** MLX's baseline also uses a K/V cache. Timing it
against a model that reruns the full prefix would compare two algorithms and
tell you nothing about where to optimize next.

### Record a Matched Baseline

Use the same model, prompt length, output length, device, and warmup count for
your solution and MLX. The required baseline uses the 0.6B model from setup:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint capacity-cache --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2 \
  --prefill-logits last

pdm run bench --solution mlx --loader week2 --model qwen3-0.6b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2 \
  --prefill-logits last
```

Swap in `--solution tiny_llm_ref` with the same arguments to compare with the
reference solution instead of MLX. Or run both rows in fresh processes:

```bash
pdm run bench-week2-progression --offline --repeats 2 \
  --solution tiny_llm --suite week2 \
  --variant week2-capacity-cache --variant mlx \
  --model qwen3-0.6b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-day1-baseline.json
```

Run benchmarks on an otherwise idle machine. Close other heavy CPU and GPU
work, keep the power mode fixed, and let the machine settle to a steady
temperature. Repeat each command and report the median. Write down the
hardware, the MLX and mlx-lm versions, the logit mode, and the exact model.
After upgrading a dependency, measure MLX again instead of reusing the old
number.

<a id="synchronize-lazy-work"></a>

### Wait for the GPU

MLX builds a computation graph lazily. If you time only the Python call, you
time building the graph, not running it. Evaluate the output inside every timed
iteration:

```python
start = perf_counter()
output = function()
mx.eval(output)
elapsed = perf_counter() - start
```

The supplied benchmark also releases each request's caches after its warmup
and timed runs, even when a run fails. If you write your own timing loop, do
the same.

<a id="attribute-the-cached-model"></a>

## Find Where the Time Goes

A benchmark tells you how fast the whole model is. To decide what to change
next, you want to know which kind of work is expensive. The supplied profiler
answers that by **replaying** one step's work through the course's public
operators, one group at a time:

```bash
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case capacity-cache:decode:128 --warmup 4 --iterations 12 \
  --json-output week2-day1-attribution.json
```

Here is what that case means. `capacity-cache:decode:128` replays one decode
step that ends with **S = 128** tokens in the cache: the first 127 tokens are
put in the cache before timing starts, and the timed step adds one more at
position 127. The profiler builds this from the loaded Qwen weights and your
solution's public functions, the same ones Week 1 and today's tasks give you.
It never builds or looks inside your `Qwen3ModelWeek2`, so it does not care
how you laid out your model, and you do not write anything extra for it. It
does need those functions to work, so run it after today's tests pass.

It times four groups, each synchronized with the GPU, and reports the median
of each:

- **projections**: every linear layer of every block, plus the output head;
- **attention**: the attention operation in every layer;
- **normalization, position, and activation**: RMSNorm, RoPE, SwiGLU, the
  embedding lookup, and the residual additions. The RoPE position offsets are
  built once per sample, inside this group's timing, and shared by every
  layer's Q and K calls;
- **KV cache update**: one append per layer through your own
  `TinyKvFullCache.update_and_fetch`, with the cache your checkpoint uses.

A few options shape the replay, and the output records all of them:

- `--cache-capacity C` sets the bound for checkpoints that use a bounded
  cache. It defaults to `C = S`, just enough for this one replayed step, which
  is smaller than the prompt-plus-output bound a real request would use.
  `C` must be at least `S`. `kv-cache` always uses an unbounded cache.
- `--prefill-logits last|all` decides how many rows go through the final norm
  and output head in a prefill case; it defaults to `last`. A decode case
  always uses one row.

The output's `replay_total_us` is the sum of the four group medians, and each
category's share is a fraction of that sum. It is not the time of a real
decode step: the groups run separately on prepared inputs, without the rest
of the model around them. The `cache` record also shows the cache's offsets,
capacity, whether the update started from an empty cache
(`first_update_from_empty`, which describes logical state, not memory
allocation), and the change in the counters from Task 1. Those counters are
your code's bookkeeping, as before, not a measure of what the GPU copied.

The replay does not check that your model uses these operators, and it is not
a whole-request timing. Those are separate checks: your tests check the
behavior they cover (outputs, offsets, capacity, and failures), reading your
model's code tells you which operators it calls, and the benchmark above
times whole requests. An earlier version of this profiler, on an M4 Pro with older
course code, put 81.5% of a `kv-cache` decode step in dense projections. Its
output is not comparable with this one's, so compare only runs you made with
the current tool and the same case, capacity, and logit setting.

Turn what you see into a plan you can test, in three sentences:

1. "Dense projections take most of the time in this replay."
2. "Keeping the weights packed in 4 bits and changing only the projection
   path should shrink that category in the replay and speed up decode in the
   whole-request benchmark."
3. "If replayed projection time does not drop, or whole-model decode gets
   slower on the same workload, I will drop or revise this idea."

Use whichever category your replay shows. The benchmark, the replay, and
this plan finish Day 1. The
[earlier macOS 27 capture lab](./week2-advanced-profiling.md) is optional
history: no trace, `gpudebug` output, screenshot, or device-specific counter gates your move to Day 2.

For an optional run with Qwen3-4B, you need enough unified memory for the
dense BF16 Week 2 model; the [model table](./preface.md#choose-a-model-for-your-mac)
suggests at least 32 GB. Download it before using `--offline`:

```bash
hf download Qwen/Qwen3-4B-MLX-4bit
```

Then rerun every compared row with `--model qwen3-4b`, and keep those results
apart from your 0.6B results; a different model is a different workload.

## Why Quantize: The Decode Roofline

If projections dominate, why are they slow? During decode, each token reads
every weight in the model once and does very little arithmetic with it, so the
speed is set by memory bandwidth, not compute. You can estimate how many bytes
that is from the
[Qwen3-4B configuration](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json).
This is a calculation for the 4B model, not a measurement of your 0.6B run:

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

The tied embedding matrix is counted once, as the output projection. The
one-row embedding lookup, normalization weights, activations, K/V reads, and
attention are left out, so this is a lower bound on bytes read and an upper
bound on speed, not a prediction. A dense FP16 or BF16 weight takes two bytes:

```plain
4,022,272,000 weights × 2 bytes = 8.045 GB per token
arithmetic intensity = 8.045 GFLOPs / 8.045 GB = 1.0 FLOP/byte
```

FP16 and BF16 split their 16 bits differently (FP16 has more precision, BF16
more range), but both take two bytes, so the bandwidth math is the same. The
course uses BF16.

| Dense weight format | Bits per weight | Bytes per weight | Streamed weight bytes per token | Weight arithmetic intensity |
|---|---:|---:|---:|---:|
| FP16 | 16 | 2 | 8.045 GB | 1.0 FLOP/byte |
| BF16 | 16 | 2 | 8.045 GB | 1.0 FLOP/byte |

One floating-point operation per byte means the GPU spends most of each decode
step waiting for weights. The checkpoint on disk already stores them in 4 bits.
[Day 2](./week2-02-quantize-model.md) keeps them that way in memory. Save your
`capacity-cache` benchmark result: it is Day 2's control.

{{#include copyright.md}}
