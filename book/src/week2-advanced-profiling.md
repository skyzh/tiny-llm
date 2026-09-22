# Optional: Read a Week 2 Evidence Package

The required Week 2 route needs only the supplied correctness gates, public
checkpoint commands, counters, and your matched product observations. This page
is an optional method for reading a preserved measurement package without
turning missing data into a claim.

The repository still contains older profiling utilities whose default cases
predate the frozen successor. They are not the runnable interface for the
current nine-checkpoint route. Use only the commands printed in the current
chapters unless a later code revision explicitly updates those utilities.

## 1. Bind Identity Before Reading Numbers

For any supplied result, record:

- exact source head and tree;
- model snapshot and model size;
- Python, MLX, and mlx-lm versions;
- device and operating system;
- prompt/output lengths, seed, warmups, and sample count;
- baseline, candidate, and whether execution order was balanced;
- which rows were excluded and why.

A timing without those fields is an observation you cannot reproduce or compare
reliably.

## 2. Separate Evidence Categories

Ask which question each record answers:

| Category | Establishes | Does not establish |
|---|---|---|
| Correctness | output is within the declared numerical contract | intended optimized route ran |
| Dispatch/counter | intended route or copy behavior occurred | latency improved |
| Component | one operator improved at one shape | complete request improved |
| Product | whole request changed under a matched control | which operator caused the change by itself |
| Source geometry | tile sizes, thread counts, static storage | occupancy, bandwidth, cache hit rate |
| Historical | why an earlier decision was plausible | a fresh result on the current head |
| Unavailable | measurement was not accepted | zero, regression, or an estimable value |

Keep these categories distinct in tables and prose.

## 3. Check the Path Before the Duration

For capacity, inspect logical-copy, physical-growth-copy, and slice-write
counters. For RMSNorm, inspect register-cached versus fixed-width-fallback
dispatches. For attention, inspect tiled-prefill, tiled-prefill-fallback, and
readable dispatches.

If the candidate never ran, the duration does not measure the mechanism. If a
fallback ran, preserve that fact rather than assigning the time to the custom
kernel.

## 4. Read Interactions Cumulatively

An independent arm answers “what happens when only this mechanism changes?” A
cumulative arm answers “what happens after the preceding mechanisms are already
enabled?” They can disagree because work shifts between phases and operators.

The accepted successor evidence therefore reports both component results and
complete-request cumulative results. Do not add the capacity, RMSNorm, and
tiled-attention percentages: their denominators and scopes differ.

## 5. Preserve Unavailable Rows

The accepted 2K/512 and 8K/128 product rows were unavailable after environmental
contamination. A valid ledger leaves them unavailable and records the reason.
It does not:

- substitute a component result;
- copy a neighboring prompt/output row;
- average partial contaminated samples;
- call the missing value zero;
- infer a long-context product verdict.

## 6. Write a Bounded Decision

Use four sentences:

1. identify the exact workload and control;
2. state the correctness and routing evidence;
3. report component and product evidence separately;
4. choose keep, reject, or inconclusive and name the falsifier.

For example: “On this fixed 2K-row operator shape, the BF16 D128 tiled prefill
path matched readable attention and its dispatch counter increased. The supplied
component median improved by 54.07%. That number is not the complete-request
gain, and the 2K/512 product row is unavailable. I keep the path only for its
eligible prefill boundary and would revisit it if a matched product row regressed
or the fallback/correctness contract failed.”

Return to the [Week 2 route](./week2-overview.md) or the
[fresh decision ledger](./appendix-performance.md#fresh-successor-decision-ledger).

{{#include copyright.md}}
