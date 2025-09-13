# Week 2 Day 6 and 7: Chunked Prefill and Continuous Batching

In this chapter, we will implement **continuous batching**. The idea is to batch multiple requests together so we can make full use of the compute resources.

So far, we have assumed that the model only processes a single batch each time it is called. However, a single batch is usually not enough to saturate the compute resources. To address this, we can process multiple requests at the same time.

The first question is how to batch requests. A naive approach would be to select a fixed number of prompts (for example, 5) from the request queue and perform decoding as before. The problem is that different prompts produce sequences of different lengths. It is possible that 4 out of 5 requests finish decoding quickly, while the remaining one takes much longer. This leads to wasted compute resources and stalls all other requests.

A smarter approach is **continuous batching**. That is, we set the maximum number of requests we can process at once. When one request finishes, we replace its slot (i.e., its KV cache) with another request. In this way, the pipeline remains fully utilized.

Another challenge is how to handle decoding and prefilling at the same time. In this chapter, we adopt a simplified approach: we prefill one request, then decode one token for each request in progress. The general idea can be described with the following pseudocode:

```python
while requests_in_queue_or_in_progress:
    if prefill_request exists:
        prefill_request.try_prefill()  # perform a chunk of chunked prefill
        if prefill_request.ready:
            if kv_cache.try_add(prefill_request):
                prefill_request = next(requests)
    tokens = decode(model, kv_cache)
    requests.append(tokens)
```

We will also implement **chunked prefill** in this chapter. Prefilling a long prompt can take a significant amount of time. Since we are interleaving prefills and decodes, we want to reduce the latency of producing the next token. Ideally, the time slots for prefill and decode should be roughly equal. To achieve this, we can prefill a portion of the request at a time, using multiple slots to finish the entire prefill.

For prefilling, this essentially means providing a chunk of tokens to the model to populate the KV cache. For example:

```python
# assume prompt_tokens is a list of 400 tokens and prefill chunk size is 128
_step(model, prompt_tokens[0:128], offset=0, kv_cache)
_step(model, prompt_tokens[128:256], offset=128, kv_cache)
_step(model, prompt_tokens[256:384], offset=256, kv_cache)
_step(model, prompt_tokens[384:400], offset=384, kv_cache)
```

Note that the causal mask generated during prefilling has the shape `LxS`. For example, assume we already have 5 tokens in the KV cache and want to prefill 3 tokens. The mask should look like this:

```
0    0    0   -inf  -inf
0    0    0    0    -inf
0    0    0    0     0
```

This is the same masking logic you implemented in Week 1.

## Task 1: Batch RoPE and Causal Mask for Prefill

```
src/tiny_llm/positional_encoding.py
src/tiny_llm/attention.py::causal_mask
```

Ensure your RoPE implementation accepts a list of offsets. Also, make sure your mask implementation correctly handles the case where `L != S`.

## Task 2: Batch KV Cache

```
src/tiny_llm/kv_cache.py::BatchingKvCache
```

The batch KV cache is a collection of KV caches, one for each request. A challenge here is generating a `BxHxLxS` mask for the batch, since requests can have different lengths.

```
S = max(S_i of the batch)
L = mask_length (input parameter)
keys: 1, H, S_i, D
values: 1, H, S_i, D
batched_keys: B, H, S, D
batched_values: B, H, S, D
mask: B, 1, L, S
```

You should fill the `batched_keys` and `batched_values` arrays so that each request’s data is aligned at the end:

```python
batched_keys[i, :, (S-S_i):S, :] = keys[i, :, :, :]
batched_values[i, :, (S-S_i):S, :] = values[i, :, :, :]
mask[i, :, 0:L, (S-S_i):S] = causal_mask(L, S_i)
```

## Task 3: Handle Batches in the Model

```
src/tiny_llm/qwen2_week2.py
```

Ensure your model can handle multiple requests simultaneously. You should also use the masks returned by the batch KV cache.

## Task 4: Batch Generate

```
src/tiny_llm/batch.py
```

Implement `try_prefill` so that it prefills an entire request at once. Then implement the rest of the code as described in the starter code.

## Task 5: Chunked Prefill

```
src/tiny_llm/batch.py
```

Modify `try_prefill` so that it performs prefilling in chunks, rather than all at once.

You can test your implementation by running:

```bash
pdm run batch-main
```

This will use the `qwen2-0.5b` model with a batch size of 5 to process a fixed set of prompts.

{{#include copyright.md}}
