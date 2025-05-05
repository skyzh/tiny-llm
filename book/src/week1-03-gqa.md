# Week 1 Day 3: Grouped Query Attention (GQA)

<div class="warning">

This book is not complete and this chapter is not finalized yet. We are still working on the reference solution, writing
tests, and unify the math notations in the book.

</div>

In day 3, we will implement Grouped Query Attention (GQA). The Qwen2 models use GQA which is an optimization technique for multi-head attention that reduces the computational and memory costs associated with the Key (K) and Value (V) projections. Instead of each Query (Q) head having its own K and V heads (like in Multi-Head Attention, MHA), multiple Q heads share the same K and V heads. Multi-Query Attention (MQA) is a special case of GQA where all Q heads share a single K/V head pair.


**Readings**

*   [GQA Paper (Training Generalized Multi-Query Transformer Models from Pre-Trained Checkpoints)](https://arxiv.org/abs/2305.13245)
*   [Qwen layers implementation in mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py)
*   [PyTorch API (the case where enable_gqa=True)](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
*   [torchtune.modules.MultiHeadAttention](https://pytorch.org/torchtune/0.3/generated/torchtune.modules.MultiHeadAttention.html)

## Task: Implement `scaled_dot_product_attention_grouped`

In this task, we will implement the grouped scaled dot product attention function, which forms the core of GQA.

Implement `scaled_dot_product_attention_grouped` in `src/tiny_llm/attention.py`. This function is similar to the standard scaled dot product attention, but handles the case where the number of query heads is a multiple of the number of key/value heads.

The main progress is the same as the standard scaled dot product attention. The difference is that the K and V heads are shared across multiple Q heads. This means that instead of having `H_q` separate K and V heads, we have `H` K and V heads, and each K and V head is shared by `n_repeats = H_q // H` Q heads.  

The core idea is to reshape `query`, `key`, and `value` so that the K and V tensors can be effectively broadcasted to match the query heads within their groups during the `matmul` operations.
    *   Think about how to isolate the `H` and `n_repeats` dimensions in the `query` tensor.
    *   Consider adding a dimension of size 1 for `n_repeats` in the `key` and `value` tensors to enable broadcasting.
Then perform the scaled dot product attention calculation (`matmul`, scale, optional mask, `softmax`, `matmul`). Broadcasting should handle the head repetition implicitly.

Note that, leverage broadcasting instead of repeating the K and V tensors is more efficient. This is because broadcasting allows the same data to be used in multiple places without creating multiple copies of the data, which can save memory and improve performance.

At last, don't forget to reshape the final result back to the expected output shape.

```
N.. is zero or more dimensions for batches
H_q is the number of query heads
H is the number of key/value heads (H_q must be divisible by H)
L is the query sequence length
S is the key/value sequence length
D is the head dimension

query: N.. x H_q x L x D
key: N.. x H x S x D
value: N.. x H x S x D
mask: N.. x H_q x L x S
output: N.. x H_q x L x D
```

You can test your implementation by running the following command:

```bash
pdm run test -k week_1_day_3_task_1 -v
```

{{#include copyright.md}}
