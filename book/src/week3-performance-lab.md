# 🚧 Week 3 Day 7 (Optional): Serving Performance Lab

> 🚧 This chapter is under review and may change.

Week 2 optimized one dense-cache request and has its own static 80%-of-MLX
acceptance gate. Week 3 changes the storage and scheduler. Its primary
benchmark must therefore exercise request turnover, incremental unknown-size
growth, dense repacking, page reuse, and fragmentation. It is acceptable for a
single static paged request to regress because page-table indirection is real.

## Keep Static and Serving Questions Separate

A static run answers whether one attention implementation is faster for one
already-allocated request. It remains a useful kernel diagnostic:

```bash
pdm run bench-course-progression --offline --repeats 3 \
  --variant week2 --variant week3 --variant mlx \
  --model qwen3-4b --input-len 2048 --output-len 129 \
  --prefill-logits last
```

That command does not test continuous admission, cache growth, fragmentation,
or page reuse. Do not use it as the Week 3 acceptance result.

## Run the Serving Progression

The paired runner generates a deterministic mixture of prompt and output
lengths, keeps four decode slots active, admits a chunked prefill alongside
them, and replaces finished requests until the queue is empty:

```bash
pdm run bench-serving-progression --offline --repeats 3 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 \
  --prefill-step 128 --warmup 1 \
  --json-output serving-qwen3-4b.json
```

Dense and paged paths run in fresh alternating processes. The warmup compiles a
complete workload. The benchmark then synchronizes and resets every page pool,
so the measured paged process starts with zero page capacity. No global maximum
context length is passed to the cache. Each request still has a generated-token
limit so the finite benchmark can complete.

The runner reports:

| Metric | What it reveals |
|---|---|
| Output tok/s and requests/s | complete scheduler throughput |
| Aggregate decode tok/s | useful work while decode slots are active |
| Peak KV bytes | dense request plus staging storage, or reserved page pools |
| Dense growth-copy bytes | old request K/V copied by concatenation |
| Dense staging-copy bytes | live K/V copied into padded batch tensors |
| Paged growth-copy bytes | old physical pages copied when a pool grows |
| Live/capacity pages | allocator occupancy and free-page reserve |
| Tail-waste slots | internal fragmentation in final live pages |
| Reused allocations | turnover served from the free list |

Copy volume is a logical count of requested data movement, not a hardware DRAM
counter. It is nevertheless matched and deterministic, and it exposes the
algorithmic copying that a static throughput number hides.

## Reference Result

On an M4 Pro with a 20-core GPU, 64 GB memory, MLX 0.32.0, and mlx-lm 0.31.3,
the median of three fresh processes was:

| Storage path | Output tok/s | Decode tok/s | Requests/s | Peak KV MiB | Avoidable KV copy MiB |
|---|---:|---:|---:|---:|---:|
| Week 2 dense KV | 35.87 | 58.65 | 0.48 | 1,096 | 209,532 |
| Week 3 paged KV | 46.70 | 104.19 | 0.62 | 576 | 504 |

Paging improves output and request throughput by 30.2%, aggregate decode by
77.7%, and lowers peak KV storage by 47.4%. Avoidable logical copy volume falls
99.8%. The paged run reaches 1,116 live pages out of 1,152 reserved, reuses
2,196 allocations, and records 15,840 unused tail slots across the layer
caches. Those are the claims this workload supports.

It does not yet test prefix sharing or speculative decoding. Add a trace with
two requests referencing the same physical prefix to measure sharing, and a
trace with accept/reject plus `rewind` events to measure speculative cache
lifecycle. Do not infer either result from the turnover benchmark.

## Continue the Measurement Loop

Vary one scheduler input at a time:

1. Change `--prefill-step` to measure throughput versus decode stalls.
2. Change `--batch-size` while keeping the queued request set fixed.
3. Expose page size as an experiment and compare tail waste with metadata cost.
4. Add per-request timestamps before making P50/P95 latency claims.

For every experiment record the hypothesis, changed variable, correctness
status, synchronized measurements, and keep/reject decision. A new kernel does
not belong in the course unless this loop identifies a remaining bottleneck
and its gain justifies the added teaching surface.

## Validate Before Measuring

```bash
pdm run test --week 3 --day 4
pdm run test --week 3 --day 5
```

The [performance appendix](./appendix-performance.md) contains the raw command,
result interpretation, and simplification ledger. Use it as a reasonableness
check, not as a substitute for measuring your machine.

{{#include copyright.md}}
