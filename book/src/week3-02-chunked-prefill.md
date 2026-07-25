# 🚧 Week 3 Day 2: Chunked Prefill

> 🚧 This chapter is under review and may change.

A long prompt can monopolize the device while active decode requests wait for
their next token. Chunked prefill gives each scheduler iteration a prompt-token
budget, limiting how long decode work can be delayed.

The scheduler policy becomes:

```plain
admit at most prefill_max_step prompt tokens
decode one token for every active request
repeat until the queue and active batch are empty
```

## Task 1: Bound Prefill Work

Update `Request.try_prefill` in `src/tiny_llm/batch.py` to select one prompt
slice, call the model with the slice's absolute offset, and mark the request
ready only after the full prompt has been processed.

```python
for start in range(0, len(prompt_tokens), prefill_max_step):
    chunk = prompt_tokens[start : start + prefill_max_step]
    model(chunk, offset=start, cache=cache)
```

The final chunk may be smaller than the configured budget. Test prompts shorter
than one chunk, exactly one chunk, and one token longer than a chunk.

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
offset. Compare chunked prefill logits with one-shot prefill logits.

## Task 3: Materialize Between Chunks

MLX is lazy. Extending an unevaluated cache repeatedly creates a long graph and
can grow memory usage. Call each layer cache's `materialize()` hook after every
chunk so the next scheduler iteration starts from materialized state. A dense
cache evaluates its key/value tuple; a paged cache evaluates the page pool
storage without first gathering it into a dense tensor.

The hook is part of the cache lifecycle rather than the scheduler's storage
logic. This lets the scheduler use dense and paged caches without inspecting
their internal representation.

## Task 4: Measure the Fairness Tradeoff

Run the same request trace with several `prefill_max_step` values. Report total
throughput and the longest interval between consecutive decode steps. Smaller
chunks usually improve fairness but add scheduler and launch overhead; choose a
default from the measured tradeoff rather than treating one chunk size as
universal.

```bash
pdm run test --week 3 --day 2
pdm run batch-main
```

{{#include copyright.md}}
