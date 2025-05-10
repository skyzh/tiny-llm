# Week 1 Day 4: RMSNorm and Multi Perceptron Layer

<div class="warning">

This book is not complete and this chapter is not finalized yet. We are still working on the reference solution, writing
tests, and unify the math notations in the book.

</div>

In day 4, we will implement two crucial components of the Qwen2 Transformer architecture: RMSNorm and the MLP (Multi-Layer Perceptron) block, also known as the FeedForward Network. RMSNorm is a layer normalization technique that helps stabilize training with less computational overhead compared to traditional layer normalization. The MLP block is a feedforward network that processes the output of the attention layers, applying non-linear transformations to enhance the model's expressiveness.


## Task 1: Implement `RMSNorm`

You will need to implement the `RMSNorm` layer in:

```
src/tiny_llm/layer_norm.py
```

**📚 Readings**

* [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
* [Qwen2 layers implementation in mlx-lm (includes RMSNorm)](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py) - See `Qwen2RMSNorm`.


RMSNorm is defined as:

$$
y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \text{weight}
$$

Where:
-   `x` is the input tensor.
-   `weight` is a learnable scaling parameter.
-   `epsilon` (eps) is a small constant added for numerical stability (e.g., 1e-5 or 1e-6).
-   `mean(x^2)` is the sum of squares and then division by the number of elements.

The normalization is applied independently to each sample’s feature vector, typically over the last dimension of input.
Note that, mean calculation should be performed with `float32` accumulation to maintain precision before taking the square root, even if the input and weights are in a lower precision format (e.g., `float16` or `bfloat16`).

```
D is the embedding dimension.

x: N.. x D
weight: D
output: N.. x D
```

You can test your implementation by running:

```bash
pdm run test -k week_1_day_4_task_1 -v
```

## Task 2: Implement the MLP Block

TBD...

{{#include copyright.md}}
