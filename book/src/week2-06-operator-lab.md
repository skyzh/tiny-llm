# 🚧 Shared-Input Fusion: Load Once for Several Projections

At `swiglu`, each projection works and the elementwise primitives are complete.
Look at the inputs to the next three calls in attention:

```text
Q = X Wqᵀ
K = X Wkᵀ
V = X Wvᵀ
```

They use different weights but the same `X`. The MLP similarly computes gate
and up projections from one hidden state. You will combine each group, first
preserving its separate outputs and then fusing the gate/up activation too.
These are two checkpoints within one shared-input experiment.

The existing learner seams are in `src/tiny_llm/week2_kernels.py`,
`src/tiny_llm/qwen3_week2.py`, and the native `week2_kernels.cpp` and `.metal`
files. Declarations and bindings are already supplied. Start by defining which
QKV inputs your implementation supports:

```bash
pdm run test --week 2 --day 6 -- -k selector
```

The selector shell initially fails. Make it report eligibility before connecting
a kernel; eligibility describes a supported operation, not a predicted speedup.

## QKV: Share the Load, Preserve Three Results

Complete `supports_shared_input_qkv` and `quantized_qkv`. The supported packed
path takes BF16 activations, unsigned packed W4 weights, and BF16 scale/bias
metadata with group size 128. Flatten the activation's leading dimensions into
rows for dispatch, then restore them on output. The supported row range is
1 through 2048, including both single-row decode and prefill tails.

All three projections must have the same input width. Their output widths can
differ: grouped-query attention generally does not give K and V the same number
of query heads as Q. Return `(Q, K, V)` with each projection's correct width and
order. A wrong split of a combined output buffer can produce plausible values
with the wrong semantic role.

Implement the native `quantized_qkv` operation using the supplied C++/Metal
shells. Load a tile of `X` for the three projections, unpack each projection's
own weights and metadata, and retain distinct accumulators and output regions.
Reuse the tiling and tail discipline from matrix prefill. Shared activation
loads do not make weight rows interchangeable.

In `Qwen3MultiHeadAttention`, select the shared operation only when
`use_shared_input_qkv` and the selector both permit it. Otherwise keep the three
separate projection calls. After projection, preserve Q/K normalization, RoPE,
cache append, attention, and the output projection exactly as before.
Increment `shared_input_qkv_dispatches` or `separate_qkv_dispatches` at the
corresponding model route so a measurement can establish which path ran.

```bash
pdm run build-ext
pdm run test --week 2 --day 6
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint shared-input-qkv --model qwen3-0.6b
```

This gate checks selector boundaries, rejection before extension dispatch,
projection results and order, and shared-versus-separate model routing. The
checkpoint table also declares the later feature flags; declaring those flags
does not mean the later operator bodies are complete.

Compare the operator on the same input shape:

```bash
pdm run bench-week2-operators --solution tiny_llm --model qwen3-0.6b \
  --section shared-input-qkv --context 32 --warmup 4 --iterations 60 \
  --json-output week2-qkv-operator.json
```

The runner compares shared and separate projections. `--context` is the row
count for this section. Repeat with `1` for decode and a tail such as `33`,
keeping both implementations on the same input.

Then compare the same product checkpoint enabled and disabled:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-shared-input-qkv --model qwen3-0.6b \
  --input-len 128 --output-len 129 --warmup 2 --prefill-logits last \
  --json-output week2-qkv-on.json

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-shared-input-qkv --model qwen3-0.6b \
  --input-len 128 --output-len 129 --warmup 2 --prefill-logits last \
  --disable-week2-shared-input-qkv --json-output week2-qkv-off.json
```

The flag disables only QKV sharing; all incoming matrix and primitive work
remains. Confirm the dispatch counters before interpreting a timing difference.
A supported row count is a reason to run the experiment, not to assume a win.

## Gate and Up: Keep the Activation Inside the Projection

Now enter from `shared-input-qkv`. The MLP computes
`SiLU(X Wgateᵀ) ⊙ (X Wupᵀ)` before its down projection. Unlike QKV, the two
intermediate projections need not be returned separately.

Implement `supports_fused_gate_up` and `quantized_gate_up_swiglu`. The packed
input contract and row range match the QKV experiment, but gate and up must
have matching weight and output shapes. Accumulate both projections in FP32,
apply SwiGLU, and write the final BF16 result. Differences in intermediate
rounding mean the fused and separate schedules need tolerance-based comparison.

Begin with just these supplied tests from the shared Day 7 file:

```bash
pdm run test --week 2 --day 7 -- -k shared_gate_up
```

The `-k` filter deliberately leaves the later dense-attention tests out. After
implementing the native operation, rebuild and repeat this slice. In `Qwen3MLP`,
route eligible enabled inputs through the shared operation and everything else
through the separate gate/up projections and existing SwiGLU. Keep the down
projection outside this fusion. Track `shared_input_gate_up_swiglu_dispatches`.

```bash
pdm run build-ext
pdm run test --week 2 --day 7 -- -k shared_gate_up
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint shared-input-gate-up-swiglu --model qwen3-0.6b

pdm run bench-week2-operators --solution tiny_llm --model qwen3-0.6b \
  --section shared-input-gate-up-swiglu --context 32 \
  --warmup 4 --iterations 60 --json-output week2-gate-up-operator.json
```

Finish with its own product control, keeping QKV sharing enabled on both sides:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-shared-input-gate-up-swiglu --model qwen3-0.6b \
  --input-len 128 --output-len 129 --warmup 2 --prefill-logits last \
  --json-output week2-gate-up-on.json

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 2 \
  --variant week2-shared-input-gate-up-swiglu --model qwen3-0.6b \
  --input-len 128 --output-len 129 --warmup 2 --prefill-logits last \
  --disable-week2-shared-input-gate-up-swiglu \
  --json-output week2-gate-up-off.json
```

Save one decision for QKV and another for gate/up. Sharing input can reduce
loads or intermediates while changing resource use and scheduling. A component
improvement can therefore coexist with an unchanged or slower request. Explain
that outcome rather than combining both fusions into one speedup claim.

The next [dense-attention chapter](./week2-07-split-k-prefill.md) starts from
`shared-input-gate-up-swiglu`. It changes how attention computes softmax and
consumes V; it does not change these projection outputs or introduce paging.

{{#include copyright.md}}
