# 🚧 Week 3 Day 1: Continuous Batching

So far, each generation loop has served one request. One decode step for one
request is a small amount of work, so most of the GPU sits idle. Today you
decode several requests in each model call.

You start from the finished
[Week 2 model](./week2-05-tiled-prefill-attention.md). Two of its pieces
already handle what batching needs: RoPE accepts one position offset per
request, and the causal mask can be rectangular. Each request still owns its
own dense K/V cache. Today you write four things around them:

- `BatchingKvCache`, which lines up several requests' caches into one batch
  and builds the matching mask;
- a switch that sends the model's quantized projections to MLX for Week 3
  (`mlx_quantized_linear`, a per-weight selector, and the
  `dispatch_week3_batch_model` factory), and passes that selector through
  `Qwen3ModelWeek2`;
- `Request.try_prefill`; and
- the loop in `batch_generate` that admits requests, decodes them together
  and retires finished ones.

The result is a **continuous batch**: several requests decode together, and a
finished request's place goes to a waiting one straight away.

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

## Task 1: Reuse RoPE and Causal Masking for Batched Requests

```
src/tiny_llm/week2_kernels.py::FastRoPE  (reuse unchanged)
src/tiny_llm/attention.py::causal_mask   (reuse unchanged)
```

Continuous batching requires one RoPE offset per batch element and a causal
mask whose query and source lengths may differ. Verify those two Week 2
interfaces before adding the scheduler so the serving layer can use one model
contract for every request position.

Verify multi-offset RoPE and the rectangular causal mask with:

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

## Task 3: Send Projections to MLX

```
src/tiny_llm/quantize.py::mlx_quantized_linear
src/tiny_llm/qwen3_week2.py::Qwen3ModelWeek2.__init__
src/tiny_llm/models.py::dispatch_week3_batch_model
```

Your Week 2 model runs its quantized projections on the kernels you wrote in
[Week 2](./week2-03-simd-matrix-prefill.md). Week 3 is about serving, not
projection kernels, so its model sends those projections to MLX instead.
Otherwise every cache and scheduler measurement this week would also carry
the remaining cost of the teaching kernels.

Add a per-weight `use_mlx_quantized_linear` selector whose default remains
`False`, preserving every Week 2 checkpoint. When selected,
`quantized_linear` should call `mx.quantized_matmul` with the same packed
weight, scales, biases, group size, bits, and transposed-weight convention.
Then implement `dispatch_week3_batch_model`, the factory that builds the
Week 3 model: the completed dense-cache Week 2 model with that selector
enabled.

Call this batch-ready model with several requests, one offset per batch
element, and the mask returned by `BatchingKvCache`. Try requests joining and
leaving at different positions. Only the quantized projections go to MLX;
normalization, RoPE, activation, attention, cache state, and scheduling stay
in your solution. The model does not know about requests: which request owns
which slot, and when it joins or leaves, belong to the cache and the
scheduler.

Verify the MLX projections and the factory with:

```bash
pdm run test --week 3 --day 1 -- -k task_3
```

## Task 4: Batch Generate

```
src/tiny_llm/batch.py
```

First implement `Request.try_prefill` by prefilling the complete prompt in one
call. The visible `prefill_max_step` input is reserved for Day 2; on Day 1, give
this call a budget that covers the complete remaining prompt. Then complete the
scheduler in `batch_generate`. Each request passes through three states:

- **Pending.** `pending_prefill_request` holds at most one request. When its
  prefill finishes, it may already be complete: its first generated token is
  EOS, or it has reached `max_seq_len`. Record its result and clear
  `pending_prefill_request` without giving it a decode slot. Otherwise find an idle slot `i`, add the request's cache for each
  layer to slot `i` of that layer's `BatchingKvCache` with `add_request`, place
  the request in `decode_requests[i]`, and clear `pending_prefill_request`. If
  every slot is busy, leave it pending. The supplied loop keeps running while a
  request is pending, so it is admitted as soon as a slot frees.
- **Decoding.** Every iteration, collect the next token and offset of each
  active slot, using a placeholder token and offset 0 for idle slots; the batch
  mask already hides them.
- **Complete.** When a slot's request reaches EOS or `max_seq_len`, call
  `remove_request(i)` on every layer's cache, record its result, and set
  `decode_requests[i]` back to `None` so the next pending request can use it.

A prompt longer than `max_seq_len` must fail before model or cache work, and a
generated token that would cross the limit is not emitted.

Results are returned in **completion order**, because short requests can leave
the batch before earlier long requests. Each result's `prompt_idx` maps it back
to the original input position; do not reorder completed results into input
order.

Use the supplied scheduler checkpoint for full prefill, admission and slot
reuse, immediate EOS, maximum-length termination, completion-order results,
and their original `prompt_idx` values:

```bash
pdm run test --week 3 --day 1 -- -k task_4
```

Then run the complete scheduler against the real model:

```bash
pdm run batch-main --solution tiny_llm --loader week2
```

By default, `batch-main` uses Qwen3-0.6B with a batch size of five and a fixed
set of prompts. Use it to watch the engine work: requests enter, decode,
finish, and give up their slots. Don't time anything with it. It shuffles the
prompt order, and its clock is cumulative.

Day 2 measures this properly with `bench-chunked-prefill`, which replays a
fixed set of requests. With a 512-token prefill budget, every prompt in its
64–512-token set fits in one chunk, which is exactly what you built today. So
a run with that budget is your Day 1 control for Day 2.

{{#include copyright.md}}
