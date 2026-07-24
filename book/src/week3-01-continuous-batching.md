# 🚧 Week 3 Day 1: Continuous Batching

> 🚧 This chapter is under review and may change.

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
        prefill_request.try_prefill()  # Day 1 processes the complete prompt
        if prefill_request.ready:
            if kv_cache.try_add(prefill_request):
                prefill_request = next(requests)
    if active_requests:
        tokens = decode(model, kv_cache)
        for request, token in zip(active_requests, tokens):
            request.append(token)
```

A complete prompt is admitted in one call on Day 1. This makes the scheduling
policy easy to inspect and exposes an important limitation: one long prefill
can delay every active request's next decode step. Day 2 will add a bounded
prefill budget to solve that fairness problem.

## Task 1: Verify the Week 2 Batch Contract

```
src/tiny_llm/week2_kernels.py::FastRoPE  (reuse unchanged)
src/tiny_llm/attention.py::causal_mask   (reuse unchanged)
```

Continuous batching requires one RoPE offset per batch element and a causal
mask whose query and source lengths may differ. Verify those two Week 2
interfaces before adding the scheduler so the serving layer can use one model
contract for every request position.

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

You can verify your solution by running:

```bash
pdm run test --week 3 --day 1 -- -k task_2
```

## Task 3: Exercise the Batch-Ready Model

```
src/tiny_llm/qwen3_week2.py  (reuse unchanged)
```

Call the Week 2 model with several requests, one offset per batch element, and
the mask returned by `BatchingKvCache`. Exercise requests joining and leaving
at different positions. The model remains request-agnostic; slot ownership and
lifecycle belong to the cache and scheduler.

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

Run the complete scheduler with:

```bash
pdm run batch-main
```

By default, this command uses Qwen3-0.6B with a batch size of five and a fixed
set of prompts. Record the longest interval between consecutive decode steps
when one queued request has a much longer prompt. That interval is the baseline
for Day 2.

{{#include copyright.md}}
