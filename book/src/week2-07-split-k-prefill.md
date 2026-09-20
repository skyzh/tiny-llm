# 🚧 I/O-Aware Dense Attention: Keep Only the State You Need

You arrive at `shared-input-gate-up-swiglu` with a dense KV cache, matrix
prefill, compact primitives, and shared-input projections. Attention still has
a useful remaining question: must it store every score before multiplying by V?

For `L` incoming query rows and `S` cached source rows, a materialized attention
implementation forms an `L × S` score array per query head, then applies softmax
and multiplies by V. You will compute the same dense attention while carrying
small running softmax state through the source positions. “Exact” here means
the same dense mathematical operation; floating-point schedules are compared
with tolerances, not bit-for-bit equality.

Start with the selector and wrapper boundary:

```bash
pdm run test --week 2 --day 7 -- -k 'io_aware_selector or io_aware_wrapper'
```

These should fail at the sparse attention functions before implementation.
Complete `should_use_io_aware_dense_attention` in
`src/tiny_llm/qwen3_week2.py` and `io_aware_dense_attention` in
`src/tiny_llm/week2_kernels.py`, then the existing native attention primitive in
`src/extensions/src/week2_kernels.cpp` and `.metal`. The supplied extension
plumbing uses the name `decode_attention`; the learner-facing wrapper and model
checkpoint are `io_aware_dense_attention` and `io-aware-dense-attention`.

## Keep the Mask and Head Mapping Explicit

The wrapper accepts rank-four tensors:

```text
Q: B, Hq, L, D
K: B, Hkv, S, D
V: B, Hkv, S, D
output: B, Hq, L, D
```

K and V shapes must match, `Hq` must be divisible by `Hkv`, and query heads must
read their corresponding KV head. The supported head dimension is 1 through
256, with `1 <= L <= S`. Q, K, and V need matching floating dtypes; the course
model uses BF16. Validate the boundary before dispatch and preserve the
model's readable fallback when the custom path is disabled or ineligible.

Support no mask, the string `"causal"`, and a broadcastable additive array
mask. Causality is aligned to the end of the cached prefix. For `L = 3` and
`S = 7`, the first incoming query can see source positions 0 through 4, the
second through 5, and the third through 6. Using `source <= query_index` would
incorrectly hide the existing prefix.

An additive mask contributes to a score before softmax. Preserve its per-row
and per-head indexing; a tensor with the correct shape but the wrong mask row
can silently compute a different answer. The supplied GPU witness includes an
explicit mask whose output differs from the unmasked control.

## Update Softmax Without Saving the Score Row

For one query, let `s` be the next scaled, masked dot product and `v` its value
vector. Maintain a running maximum `m`, exponential sum `l`, and unnormalized
weighted value vector `a`. When the maximum changes, rescale the old state:

$$
\begin{aligned}
m' &= \max(m,s),\\
\alpha &= \exp(m-m'),\qquad \beta = \exp(s-m'),\\
l' &= \alpha l + \beta,\\
a' &= \alpha a + \beta v.
\end{aligned}
$$

After all visible source positions, the output is `a / l`. Handle empty partial
work explicitly when initializing or merging state. If you forget to rescale
`a` when a larger score arrives, the denominator and numerator describe
different softmax distributions.

Parallel workers can cover disjoint source positions and merge their states.
For partial states `(mj, lj, aj)`, choose the largest partial maximum `m*`,
rescale every `lj` and `aj` by `exp(mj - m*)`, and sum. Divide only after that
merge. An average of already-normalized partial outputs gives the partitions
equal weight even when their softmax masses differ.

Accumulate scores and online state in FP32 and cast the final model output to
BF16. The kernel avoids materializing the score/probability arrays. It still
reads dense K/V, and an explicitly supplied mask can itself occupy `L × S`
space. Avoid describing the whole operation as having no quadratic input or
storage under every mask representation.

## Reach the Final Model Checkpoint

In `Qwen3MultiHeadAttention`, select the new wrapper only when the feature is
enabled and its selector accepts the inputs. Otherwise use the readable dense
attention path. Preserve the cache update, projection results, mask, scale, and
output shape. Track `io_aware_dense_attention_dispatches` and
`readable_attention_dispatches` so the chosen route is observable.

Once the native implementation is connected, run its GPU comparison, then the
complete Day 7 gate, which includes the preceding gate/up checkpoint:

```bash
pdm run build-ext
pdm run test --week 2 --day 7 -- -k io_aware
pdm run test --week 2 --day 7
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint io-aware-dense-attention --model qwen3-0.6b
```

The gate compares results against materialized dense attention and checks mask,
shape, dtype, selector, and wrapper behavior. The complete prompt call exercises
the final model route. Keep the earlier shared-input checkpoint runnable as an
additional control.

## Compare the Same Attention, Then the Same Request

The operator runner can vary source length and query length separately:

```bash
pdm run bench-week2-operators --solution tiny_llm --model qwen3-0.6b \
  --section attention --context 128 --query-length 1 \
  --attention-mask none --warmup 4 --iterations 60 \
  --json-output week2-attention-decode.json

pdm run bench-week2-operators --solution tiny_llm --model qwen3-0.6b \
  --section attention --context 128 --query-length 32 \
  --attention-mask causal --warmup 4 --iterations 60 \
  --json-output week2-attention-prefill.json
```

Each run contains matched implementation comparisons for that shape. The two
commands ask different questions; do not subtract their times and call the
difference a speedup. Use these small shapes first, then vary context and query
length deliberately.

Compare complete requests with only dense attention toggled:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-io-aware-dense-attention --model qwen3-0.6b \
  --input-len 128 --output-len 129 --warmup 2 --prefill-logits last \
  --json-output week2-attention-on.json

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-io-aware-dense-attention --model qwen3-0.6b \
  --input-len 128 --output-len 129 --warmup 2 --prefill-logits last \
  --disable-week2-io-aware-dense-attention \
  --json-output week2-attention-off.json
```

QKV and gate/up sharing remain enabled on both sides. Confirm dispatch, check
correctness, and compare prefill and decode independently. Avoid promoting an
operator result into a complete-request claim when those measurements disagree.

## Finish with Decisions You Can Defend

Complete the [decision ledger](./week2-decision-ledger.md) for every change,
including results you rejected or could not distinguish from noise. An eligible,
correct custom path is not an obligation to deploy it on every shape. Your
record should say which workload supports the choice and what would reverse it.

You have now optimized the work inside one request. A growing dense cache still
belongs to that request, and dense attention still addresses it as a contiguous
logical sequence. [Week 3](./week3-overview.md) adds scheduling and a paged cache
layout. Carry forward the cache lifecycle, offsets, grouped-head mapping, and
mask semantics; do not assume the dense kernel already knows how to follow a
page table.

{{#include copyright.md}}
