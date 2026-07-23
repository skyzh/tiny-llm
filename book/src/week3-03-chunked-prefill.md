# 🚧 Week 3 Day 3: Chunked Prefill

> 🚧 This newly introduced chapter is a work in progress.

A long prompt can monopolize the device while active decode requests wait for
their next token. Chunked prefill limits the number of prompt tokens admitted in
one scheduler step.

```python
for start in range(0, len(prompt_tokens), prefill_max_step):
    chunk = prompt_tokens[start : start + prefill_max_step]
    model(chunk, offset=start, cache=cache)
```

## Rectangular Causal Masks

When a cache already holds `S - L` tokens and a chunk contributes `L` new
tokens, the mask is `L x S`. Every query can attend to the old prefix and to
earlier positions in its own chunk.

## Materialize Between Chunks

MLX is lazy. Extending an unevaluated cache repeatedly creates a long graph and
can grow memory usage. Evaluate every layer's key and value tensors after each
chunk so the next scheduler iteration starts from materialized state.

## Task: Bound Prefill Work

Update `Request.try_prefill` in `src/tiny_llm/batch.py` to process at most
`prefill_max_step` tokens, advance its offset, materialize the cache, and mark
the request ready only when the full prompt is complete.

```bash
pdm run batch-main
```

Compare time-to-next-token for active decode requests with a small and a large
prefill step. Smaller chunks improve fairness but add scheduling overhead.

{{#include copyright.md}}
