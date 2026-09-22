# 🚧 Week 2 Day 2: Bound KV-Cache Movement

The `kv-cache` checkpoint stops recomputing old tokens, but its readable
concatenation still copies the old K/V prefix on every append. Day 2 separates
two quantities that concatenation conflates:

- **logical length**: tokens currently visible to attention;
- **physical capacity**: token slots owned by the request.

Your seam is `TinyKvFullCache(capacity=...)` plus the request-budget allocation
in the generation path. The optimized cache allocates once, writes only the new
slice, and returns a view of the logical prefix.

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

## Measure and Decide

Use identical prompts and output bounds for the two public checkpoints:

```bash
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-0.6b --max-tokens 16
/usr/bin/time -p pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint capacity-cache --model qwen3-0.6b --max-tokens 16
```

Record both the coarse product observation and the copy counters. Keep the
mechanism when it preserves logical behavior, makes overflow/reset/rewind
correct, and removes repeated prefix copies within the declared request bound.
Treat a noisy time difference as inconclusive rather than contradicting the
counter witness.

A historical exact-mechanism run observed a **+88.0 MiB / +2.276%** temporal
2K/512 peak-memory tradeoff. That is historical evidence, not a result from the
current frozen head. Capacity exchanges repeated copy work for up-front bounded
storage; it does not promise lower peak memory at every request shape.

Continue to [packed W4 projection weights](./week2-03-quantize-model.md).

{{#include copyright.md}}
