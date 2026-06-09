# Week 3 Day 3: Mixture of Experts

In this chapter, we will implement the feed-forward shape of **Mixture of
Experts**, or **MoE**, for the Qwen3 family.

So far, every transformer block in tiny-llm has used the same dense Qwen3 MLP:

```plain
x -> gate_proj
x -> up_proj
SiLU(gate_proj(x)) * up_proj(x) -> down_proj
```

That is a SwiGLU MLP. Every token visits the same weights.

MoE changes only the feed-forward half of the transformer block. Instead of one
dense MLP, the model owns many expert MLPs. A small router chooses which experts
each token should use:

```plain
token hidden state -> router -> top-k experts -> weighted expert outputs
```

The attention path does not change. KV cache does not change. The sparse work is
inside the MLP half of the block.

**Readings**

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

## Dense MLP vs MoE MLP

The dense Qwen3 MLP from Week 1 has one set of weights:

```plain
w_gate: hidden_dim, dim
w_up:   hidden_dim, dim
w_down: dim, hidden_dim
```

A Qwen3-MoE sparse block has a bank of those weights:

```plain
expert_gate: num_experts, moe_hidden_dim, dim
expert_up:   num_experts, moe_hidden_dim, dim
expert_down: num_experts, dim, moe_hidden_dim
```

The router produces one score per expert:

```plain
router_logits: B, L, num_experts
router_probs:  softmax(router_logits)
```

Then the model picks `num_experts_per_tok` experts for each token:

```plain
expert_ids:    B, L, num_experts_per_tok
expert_scores: B, L, num_experts_per_tok
```

For each token, only those selected experts run. Their outputs are weighted and
summed:

```plain
output[token] = sum(score_i * expert_i(token))
```

That is the central MoE idea: the model can contain many parameters, but each
token activates only a small subset of them.

## Qwen3-MoE Shape

Qwen3-MoE keeps the same attention structure as Qwen3, including QK norm, GQA,
RoPE, and the same KV cache interface. It replaces some dense MLP layers with a
sparse MoE block.

The useful pieces are:

- `gate`: a router linear layer from hidden size to `num_experts`
- `switch_mlp`: many SwiGLU experts with `moe_intermediate_size`
- `num_experts_per_tok`: how many experts a token uses
- `norm_topk_prob`: whether selected expert scores are renormalized
- `decoder_sparse_step` and `mlp_only_layers`: which layers are sparse vs dense

There is no shared expert in the Qwen3-MoE block we are following. The sparse
feed-forward output is just the weighted top-k expert mixture.

## Grouped Quantized Matmul

MLX does not give us a single high-level MoE block in `mlx.nn`. It does have a
lower-level primitive, `mx.gather_qmm`, that performs quantized matrix
multiplication while selecting a different matrix for each row. In this chapter,
we will build a narrow teaching version of that idea:
`grouped_quantized_matmul`.

For MoE, that means:

```plain
token rows:  N, D
expert ids:  N
weights:     E, O, D packed as 4-bit QuantizedWeights
output:      N, O
```

The row with `expert_ids[i] = e` should multiply by `weights[e]`.

Task 1 will assume the rows are already sorted by expert id. The MoE helper will
keep the inverse order from the sort so the result can be restored to the
original token order.

## Router Step

The router is just a quantized linear layer:

```python
router_logits = quantized_linear(x, w_router)
router_probs = softmax(router_logits, axis=-1)
```

For a batch of tokens:

```plain
x:             B, L, D
router_logits: B, L, E
router_probs:  B, L, E
```

where `E = num_experts`.

Qwen3-MoE then uses top-k selection:

```python
expert_ids = argpartition(-router_probs, k)[:k]
expert_scores = take_along_axis(router_probs, expert_ids)
```

If `norm_topk_prob` is true, renormalize `expert_scores` so the selected scores
sum to 1 for each token.

## Expert Step

Each expert is the same kind of SwiGLU MLP we already know:

```plain
expert(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

The implementation should build token-expert jobs, group them by expert, and run
the expert projections with `grouped_quantized_matmul`:

```plain
selected expert ids -> expanded token-expert rows
expanded rows -> sort/group by expert id
grouped expert rows -> grouped gate/up projection
SiLU(gate) * up -> grouped down projection
restore original token/top-k order -> weighted sum
```

The reorder is part of the model implementation. It keeps all token rows for the
same expert contiguous so the expert bank can be applied with grouped matrix
multiplication.

## Task 1: Grouped Quantized Matmul

```
src/extensions/src/quantized_matmul.cpp
src/extensions/src/quantized_matmul.metal
src/tiny_llm/quantize.py
src/tiny_llm/moe.py
```

Implement `grouped_quantized_matmul`, then use it from `grouped_expert_linear`.
This is the quantized grouped-matmul core of MoE.

`grouped_quantized_matmul` accepts:

```plain
a:           R, D
w_experts:   packed QuantizedWeights for num_experts, output_dim, D
expert_ids:  R, sorted by expert id
```

It returns:

```plain
out:         R, output_dim
```

Each row uses the expert selected by the matching row in `expert_ids`:

```plain
out[row] = a[row] @ dequantize(w_experts[expert_ids[row]]).T
```

The implementation should:

```plain
1. add a Python wrapper for grouped_quantized_matmul,
2. extend the quantized matmul extension with a grouped entrypoint,
3. read expert_ids[row] inside the kernel,
4. use that expert id to choose the expert weight, scale, and bias row.
```

After that, implement `grouped_expert_linear` in `src/tiny_llm/moe.py`:

```plain
1. flatten token rows and expert ids,
2. sort rows by expert id,
3. call grouped_quantized_matmul,
4. restore the original order.
```

The call should look like:

```python
out = grouped_quantized_matmul(
    w_experts.scales,
    w_experts.biases,
    group_size=w_experts.group_size,
    bits=w_experts.bits,
    a=grouped_rows,
    b=w_experts.weight,
    expert_ids=grouped_expert_ids,
    transpose_b=True,
)
```

This task maps to the same idea as `QuantizedSwitchLinear` in `mlx-lm`: each
token row uses a different packed expert matrix, and the expert ids choose the
right matrix.

## Task 2: Router Top-k

```
src/tiny_llm/moe.py
```

Implement `route_topk`. It accepts hidden states and router weights, then
returns:

- router probabilities
- selected expert ids
- selected expert scores

Use `quantized_linear` and `softmax`. Use `mx.argpartition` to select the top
`num_experts_per_tok` experts, then `mx.take_along_axis` to gather their scores.

Keep `norm_topk_prob` as an argument because Qwen3-MoE stores this behavior in
the model config.

## Task 3: Qwen3 Sparse MoE Block

```
src/tiny_llm/moe.py
```

Implement `Moe` by composing Task 1 and Task 2:

```plain
hidden states -> route_topk
hidden states + expert ids -> grouped gate projection
hidden states + expert ids -> grouped up projection
SiLU(gate) * up -> grouped down projection
weighted sum over num_experts_per_tok
```

This completes the Qwen3-MoE sparse feed-forward block. There is no shared expert
branch in this block.

## Task 4: Integrate Qwen3-MoE Layers

```
src/tiny_llm/qwen3_week3.py
src/tiny_llm/models.py
```

Add a Qwen3-MoE loader path that reuses the Week 3 Qwen3 attention and paged KV
cache behavior, but swaps selected block MLPs for `Moe`.

The model wrapper should:

- keep Qwen3 attention unchanged,
- use regular `Qwen3MLP` for `mlp_only_layers`,
- use `Moe` for sparse layers selected by
  `decoder_sparse_step`,
- load router and expert weights as `QuantizedWeights` from the Qwen3-MoE MLX
  model,
- preserve the same decode call shape:

```python
logits = model(tokens, offset, cache)
```

No scheduler API change in `src/tiny_llm/batch.py` is required for correctness.

Run this task through the normal generation entrypoints instead of adding a
separate unit test. For example:

```bash
hf download Qwen/Qwen3-30B-A3B-MLX-4bit

pdm run main --solution tiny_llm --loader week3 --model qwen3-30b-a3b \
  --prompt "Give me a short introduction to mixture of experts."

pdm run batch-main --solution tiny_llm --loader week3 --model qwen3-30b-a3b \
  --batch-size 2 --prefill-step 16
```

{{#include copyright.md}}
