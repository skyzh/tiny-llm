# Week 3 Day 1: Continuous Batching

In this chapter, we will implement **continuous batching**, which keeps a batch
of active requests on the device and replaces each request as soon as it
finishes.

So far, each generation loop has processed only one request. That may not provide
enough work to use the device efficiently, so we will decode several requests
in each model call.

A static batch could select five prompts and run them together until every
request finishes. However, generated sequences have different lengths. If four
requests finish quickly while the fifth continues, most of the batch remains
idle and queued requests cannot start.

Continuous batching instead sets a maximum number of active decode requests.
When one finishes, the scheduler assigns its batch slot and KV-cache entry to a
waiting request. This keeps the decode batch populated whenever work is queued.

The scheduler must also interleave prefill and decode work. We will use a simple
policy: advance one pending prefill, then decode one token for every active
request.

```python
while requests_in_queue_or_in_progress:
    if prefill_request is not None:
        prefill_request.try_prefill()  # perform a chunk of chunked prefill
        if prefill_request.ready:
            if kv_cache.try_add(prefill_request):
                prefill_request = next(requests)
    if active_requests:
        tokens = decode(model, kv_cache)
        for request, token in zip(active_requests, tokens):
            request.append(token)
```

Day 3 will refine this scheduler with **chunked prefill**. A long prompt can make one prefill
step much slower than a decode step, delaying every active request's next token.
Splitting the prompt into smaller chunks bounds the amount of prefill work in
each scheduler iteration.

Each chunk adds another range of prompt tokens to the request's KV cache:

```python
# prompt_tokens contains 400 tokens; the chunk size is 128
_step(model, prompt_tokens[0:128], offset=0, kv_cache)
_step(model, prompt_tokens[128:256], offset=128, kv_cache)
_step(model, prompt_tokens[256:384], offset=256, kv_cache)
_step(model, prompt_tokens[384:400], offset=384, kv_cache)
```

The causal mask for each chunk has shape `L x S`, where `L` is the chunk length
and `S` is the total sequence length after appending the chunk. For example, if
the cache already contains five tokens and the next chunk contains three, the
mask has shape `3 x 8`:

```
0  0  0  0  0  0  -inf  -inf
0  0  0  0  0  0     0  -inf
0  0  0  0  0  0     0     0
```

Each row can attend to all five cached tokens, itself, and any earlier token in
the same chunk.

## Task 1: Verify the Week 2 Batch Contract

```
src/tiny_llm/week2_kernels.py::FastRoPE  (reuse unchanged)
src/tiny_llm/attention.py::causal_mask   (reuse unchanged)
```

Week 3 begins by exercising interfaces established earlier rather than editing
them retroactively. Confirm that Week 2 `FastRoPE` accepts one integer offset
per batch element and that the existing causal-mask helper handles `L != S`,
as required by chunked prefill. If either contract is missing, return to the
corresponding earlier-week task and complete it there; do not create a second
incompatible implementation in Week 3.

Verify multi-offset RoPE and both attention paths with:

```bash
pdm run test --week 3 --day 1 -- -k task_1
```

## Task 2: Batch KV Cache

```
src/tiny_llm/kv_cache.py::BatchingKvCache
```

`BatchingKvCache` holds one request cache per decode slot. Because requests may
have different sequence lengths, it must combine their keys and values into
dense tensors and construct a matching `B x 1 x L x S` mask.

```
S = max(S_i across active requests)
L = mask_length (input parameter)
request_keys: H, S_i, D
request_values: H, S_i, D
batched_keys: B, H, S, D
batched_values: B, H, S, D
mask: B, 1, L, S
```

Right-align each active request in the common `S` dimension. The leading
positions remain zero and masked out. Inactive slots remain fully masked.

```python
keys_i, values_i = request_cache[i]
batched_keys[i, :, (S - S_i):S, :] = keys_i
batched_values[i, :, (S - S_i):S, :] = values_i
mask[i, :, 0:L, (S - S_i):S] = causal_mask(L, S_i)
```

You can verify your implementation by running:

```bash
pdm run test --week 3 --day 1 -- -k task_2
```

## Task 3: Exercise the Batch-Ready Model

```
src/tiny_llm/qwen3_week2.py  (reuse unchanged)
```

The Week 2 model already accepts multiple requests, a separate offset for each
batch element, and the mask returned by `BatchingKvCache`. Exercise that
contract with several requests joining and leaving at different positions.
The new Week 3 work belongs in the cache and scheduler; do not modify the Week
2 model to make this test pass.

You should pass all of the tests by running:

```bash
pdm run test --week 3 --day 1 -- -k task_3
```

## Task 4: Batch Generate

```
src/tiny_llm/batch.py
```

First implement `Request.try_prefill` by prefilling the complete prompt in one
call. Then complete the scheduler in `batch_generate`: move finished prefills
into idle decode slots, collect the next token and offset for each slot, and
remove requests that reach EOS or `max_seq_len`.

## Day 3 Preview: Chunked Prefill

```
src/tiny_llm/batch.py
```

On Day 3, modify `Request.try_prefill` to process at most `prefill_max_step`
prompt tokens per call.

Materialize the KV cache between chunks. MLX evaluates lazily, so repeatedly
extending an unevaluated cache creates an increasingly long computation graph
and allows memory usage to grow. Calling `mx.eval` on every layer's key and
value tensors after each chunk stores the current cache and truncates that
graph.

You can test your implementation by running:

```bash
pdm run batch-main
```

By default, this command uses Qwen3-0.6B with a batch size of five and a fixed
set of prompts.

{{#include copyright.md}}
