# 🚧 Week 3 Day 2: Chunked Prefill

> 🚧 This chapter is under review and may change.

On Day 1, a new request's whole prompt is prefilled in one go. While a
2,000-token prompt is being prefilled, every request already in the batch waits
for its next token. Chunked prefill gives each pass of the scheduler a budget
of prompt tokens, so a long prompt is processed a chunk at a time, and the
requests already decoding get a turn between chunks.

The scheduler policy becomes:

```plain
admit at most prefill_max_step prompt tokens
decode one token for every active request
repeat until the queue and active batch are empty
```

## Task 1: Bound Prefill Work

Update `Request.try_prefill` in `src/tiny_llm/batch.py` so that each call
processes one slice of the prompt, at most `prefill_max_step` tokens long,
starting where the previous call stopped. Call the model with the slice's
absolute position in the prompt, and mark prefill done only after the last
slice:

```python
# one call to try_prefill
start = self.offset
chunk = self.prefill_tokens[start : start + self.prefill_max_step]
# run the model on `chunk` at offset `start`, then advance self.offset
# set self.is_prefill_done once self.offset reaches the prompt length
```

The final chunk may be smaller than the configured budget. Test prompts shorter
than one chunk, exactly one chunk, and one token longer than a chunk.

Day 2's supplied tests are named by behavior rather than task number. This one
checks chunk boundaries against one-shot prefill, together with Task 3's
materialization:

```bash
pdm run test --week 3 --day 2 -- -k chunk_boundaries
```

With chunked prefill in place, the Day 1 loop already interleaves: a long
prompt advances one chunk per iteration while active requests keep decoding.
The test for that order, `-k active_decode_advances`, also checks that every
request cache is released, so it turns green with Task 4.

## Task 2: Build Rectangular Causal Masks

When a cache already holds `S - L` tokens and a chunk contributes `L` new
tokens, the mask is `L x S`. Every query can attend to the old prefix and to
earlier positions in its own chunk.

For a five-token prefix and a three-token chunk, the mask is `3 x 8`:

```plain
0  0  0  0  0  0  -inf  -inf
0  0  0  0  0  0     0  -inf
0  0  0  0  0  0     0     0
```

Use the absolute cache offset for RoPE and `S - L` as the causal diagonal
offset. Chunked prefill should then produce the same logits as one-shot
prefill. The focused test below does not run the model; it checks the mask
itself. `causal_mask(L=2, S=5)` must keep the earlier positions visible, and
attending through that mask must give the final query the same result as
attending over the whole sequence at once:

```bash
pdm run test --week 3 --day 2 -- -k rectangular_mask
```

## Task 3: Materialize Between Chunks

MLX is lazy. Extending an unevaluated cache repeatedly creates a long graph and
can grow memory usage. Call each layer cache's `materialize()` hook after every
chunk so the next scheduler iteration starts from materialized state. For the
dense cache, implement `TinyKvFullCache.materialize` to evaluate its key/value
tuple. The paged cache that arrives on Day 3 will evaluate its page pool
storage without first gathering it into a dense tensor.

The hook is part of the cache lifecycle rather than the scheduler's storage
logic. This lets the scheduler use dense and paged caches without inspecting
their internal representation.

## Task 4: Release Every Request Cache

Each request owns one cache per layer from the moment `Request` is constructed.
The scheduler must release each of those caches exactly once, however the
request ends:

- A request that completes during prefill, from an immediate EOS or by
  reaching `max_seq_len`, never enters a slot: release its caches directly.
- A request that completes in a slot is released by `remove_request(i)`, so
  your `BatchingKvCache.remove_request` must call the request cache's
  `release()` before clearing the slot.
- A prompt longer than `max_seq_len` never gets caches: Day 1 already rejects
  it before `create_kv_cache()` is called, and its Task 4 test still requires
  that.
- If prefill, materialization, decoding or detokenization raises, the error
  should still reach the caller, but first release every cache that is still
  live, whether it belongs to the pending request or to a slot. Wrap the
  scheduler loop in `try`/`finally` for this. A cache can be referenced from
  both places while a request is being admitted, so release each cache object
  only once.

These tests count live caches through normal completion, an oversized prompt
and an injected failure at each of those four points. Then rerun the
interleaving test from Task 1, which also checks release:

```bash
pdm run test --week 3 --day 2 -- -k "release or oversized"
pdm run test --week 3 --day 2 -- -k active_decode_advances
```

## Task 5: Measure the Fairness Tradeoff

Run the same request trace with several `prefill_max_step` values. Report total
throughput and the longest interval between consecutive decode steps. Smaller
chunks usually improve fairness but add scheduler and launch overhead; choose a
default from the measured tradeoff rather than treating one chunk size as
universal.

```bash
week3_result_root="$HOME/tiny-llm-week3-results"
mkdir -p "$week3_result_root"
week3_result_dir="$(mktemp -d "$week3_result_root/run-XXXXXX")"

pdm run test --week 3 --day 2
pdm run batch-main --solution tiny_llm --loader week2

pdm run bench-chunked-prefill --solution tiny_llm --offline --model qwen3-0.6b \
  --prefill-steps 32 128 512 --num-seqs 8 --batch-size 4 \
  --min-input-len 64 --max-input-len 512 \
  --min-output-len 32 --max-output-len 32 \
  --warmup 1 --repeats 4 --cooldown-seconds 1 \
  --json-output "$week3_result_dir/chunked-prefill-tiny-llm.json"
```

This measures your `tiny_llm` solution and writes a new JSON file. To compare
with the reference solution, rerun the same flags with `--solution ref` and a
different file name.

For a sense of what to expect, the table below comes from an earlier run on an
M4 Pro, made before Week 2 was rebuilt, so your numbers will differ. It used
the same kind of trace: a fixed seed, a 32-token output budget for every
request, and each chunk size run four times in fresh processes, twice in each
order. Only the prefill budget changed between rows; the projections went to
MLX, and the scheduler, cache and attention were the course's own.

The last column is the one fairness is about. A **decode gap** is the wall
time between two consecutive decode calls while at least one request is
decoding, so it includes any prefill and scheduler work done in between. Time
with no request decoding is not counted. The medians over the four processes
were:

| Prefill budget | Output tok/s | Prefill tok/s | Decode tok/s | Requests/s | Decode step p95 | Decode gap p95 / max |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 105.23 | 2,549.62 | 181.77 | 3.288 | 15.82 ms | 30.01 / 52.62 ms |
| 128 | 153.82 | 4,215.12 | 242.23 | 4.807 | 17.79 ms | 45.36 / 53.76 ms |
| 512 | 170.46 | 4,769.14 | 262.01 | 5.327 | 17.11 ms | 73.56 / 119.90 ms |

The 512-token row was the full-prompt Day 1 control for that trace. Relative to
that row, the 128-token budget gives up 9.8% output throughput while reducing
the p95 completion gap by 38.3% and the maximum gap by 55.2%. The 32-token
budget reduces the p95 gap further but gives up substantially more throughput.
On that workload, 128 was a reasonable compromise; on yours, the best value
may differ. The [performance appendix](./appendix-performance.md) has the full
record of the older run.

{{#include copyright.md}}
