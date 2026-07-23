# 🚧 Week 3 Day 7 (Optional): Serving Performance Lab

> 🚧 This chapter is under review and may change.

Week 2 optimized the model operators. Week 3 added request scheduling and paged
KV ownership. This lab measures the serving system as a system: batch size,
prefill chunk size, page size, request mix, allocator reuse, and fairness. It
does not introduce another matrix kernel or change the model dtype.

## Start from the Completed Interfaces

The following boundaries are prerequisites, not tasks for this lab:

- Week 2 quantized linear dispatches SIMD matvec for decode and SIMD-matrix
  matmul for prefill.
- Week 3 paged FlashAttention handles long prefill.
- Generation requests only the final logit row.
- Week 3 continuous batching owns request admission and slot reuse.
- Week 3 paged caches own physical K/V pages and expose page metadata to
  paged attention.

If one of those paths is missing, return to its chapter. Do not hide an
incomplete prerequisite behind a benchmark-only workaround.

## Choose Serving Metrics

One number cannot describe a serving engine. Record at least:

| Metric | What it reveals |
| --- | --- |
| Time to first token | prompt admission and prefill delay |
| Time between tokens | decode latency and scheduler stalls |
| Aggregate decode tok/s | useful work across active requests |
| Request throughput | admission, completion, and slot reuse |
| Peak KV pages | cache capacity and fragmentation |
| P50/P95 latency | average behavior and tail fairness |

Keep the workload fixed while varying one scheduling choice. Record model,
MLX version, hardware, prompt/output distribution, warmup, and repeat count.

For a single-request serving baseline against MLX, request only the final
prompt logit row and exclude Week 1, whose readable interface always computes
all rows:

```bash
pdm run bench-course-progression --offline --repeats 3 \
  --variant week2 --variant week3 --variant mlx \
  --prefill-logits last --input-len 512 --output-len 65
```

Use the default `--prefill-logits all` only when the workload really consumes
the score at every prompt position. It is a matched prompt-scoring benchmark,
not the generation-serving prefill path.

## Experiment 1: Prefill Chunk Size

Run the same mixture of long prompts and active decoders with several
`prefill_max_step` values. A smaller chunk bounds scheduler stalls but adds more
model calls and cache updates. Compare aggregate throughput and P95
time-between-tokens; neither metric alone is sufficient.

```bash
pdm run bench --solution tiny_llm_ref --loader week3 --batch-decode \
  --num-seqs 16 --batch-size 4 --prefill-step 32 --model qwen3-0.6b

pdm run bench --solution tiny_llm_ref --loader week3 --batch-decode \
  --num-seqs 16 --batch-size 4 --prefill-step 128 --model qwen3-0.6b
```

## Experiment 2: Batch Size and Request Turnover

Increase active batch size while keeping the queued request set fixed. Verify
that finished requests release their pages and that newly admitted requests
reuse capacity. Report aggregate throughput together with per-request latency;
a larger batch can improve the former while worsening the latter.

## Experiment 3: Page Size

Compare several page sizes with the same sequence-length distribution. Small
pages reduce unused tail capacity but lengthen block tables and increase page
management. Large pages shorten metadata but waste more space at request tails.

Measure page count, unused tail slots, and attention latency. Do not call a page
size better merely because one isolated kernel becomes faster.

## Experiment 4: Dense and Paged Attention

Use the dense-gather compatibility path from Day 3 as an ablation against the
direct paged path from Day 4. They must run the same requests and reuse the same
Week 2 model operators. Attribute differences to gather/repack work, page-table
indirection, and cache reuse rather than to unrelated prefill kernels.

Long prefill should run Week 3 paged FlashAttention directly from page storage.
A later chunk with cached history uses paged prefill. Short decode uses the
paged vector schedule. Measure these shapes separately before combining them in
one serving workload.

## Preserve an Optimization Ledger

For each experiment record:

1. the hypothesis;
2. the single changed variable;
3. correctness status;
4. synchronized measurements;
5. the keep or reject decision and its tradeoff.

Fewer graph nodes, fewer page-table entries, or more active requests are
hypotheses. Only matched measurements reveal their effect on latency,
throughput, memory, and fairness.

## Validate Before Measuring

Run the completed serving tests before collecting a performance result:

```bash
pdm run test --week 3 --day 4
pdm run test --week 3 --day 5
```

The [performance appendix](./appendix-performance.md) contains representative
results and the retained optimization ledger. Use it as a reasonableness check,
not as a substitute for measuring your machine.

{{#include copyright.md}}
