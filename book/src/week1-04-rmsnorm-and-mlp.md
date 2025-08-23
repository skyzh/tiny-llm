# Week 1 Day 4: RMSNorm and Multi Perceptron Layer

In day 4, we will implement two crucial components of the Qwen2 Transformer architecture: RMSNorm and the MLP (Multi-Layer Perceptron) block, also known as the FeedForward Network. RMSNorm is a layer normalization technique that helps stabilize training with less computational overhead compared to traditional layer normalization. The MLP block is a feedforward network that processes the output of the attention layers, applying non-linear transformations to enhance the model's expressiveness.


## Task 1: Implement `RMSNorm`

In this task, we will implement the `RMSNorm` layer.

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
pdm run test --week 1 --day 4 -- -k task_1
```

## Task 2: Implement the MLP Block

In this task, we will implement the MLP block named `Qwen2MLP`.

```
src/tiny_llm/qwen2_week1.py
```

The original Transformer model utilized a simple Feed-Forward Network (FFN) within each block. This FFN typically consisted of two linear transformations with a ReLU activation in between, applied position-wise.

Modern Transformer architectures, including Qwen2, often employ more advanced FFN variants for improved performance. Qwen2 uses a specific type of Gated Linear Unit (GLU) called SwiGLU.

**📚 Readings**
* [Attention is All You Need (Transformer Paper, Section 3.3 "Position-wise Feed-Forward Networks")](https://arxiv.org/abs/1706.03762)
* [GLU Paper(Language Modeling with Gated Convolutional Networks)](https://arxiv.org/pdf/1612.08083)
* [SilU(Swish) activation function](https://arxiv.org/pdf/1710.05941)
* [SwiGLU Paper(GLU Variants Improve Transformer)](https://arxiv.org/abs/2002.05202v1)
* [PyTorch SiLU documentation](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
* [Qwen2 layers implementation in mlx-lm (includes MLP)](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py)

Essentially, SwiGLU is a combination of GLU and the SiLU (Sigmoid Linear Unit) activation function:
-  GLU is a gating mechanism that allows the model to learn which parts of the input to focus on. It typically involves an element-wise product of two linear projections of the input, one of which might be passed through an activation function. Compared to ReLU used in the original FFN, GLU can help the model learn more complex relationships in the data, deciding which features to keep and which to discard.
-  SiLU (Sigmoid Linear Unit) is a smooth, non-monotonic activation function that has been shown to perform well in various deep learning tasks. Compared to ReLU and sigmoid used in GLU, it is fully differentiable without the zero-gradient “dead zones”, retains non-zero output even for negative inputs.

You need to implement the `silu` function in `basics.py` first. For `silu`, it takes a tensor of the shape `N.. x I` and returns a tensor of the same shape.
The `silu` function is defined as:
$$
\text{SiLU}(x) = x * \text{sigmoid}(x) = \frac{x}{1 + e^{-x}}
$$


Then implement `Qwen2MLP`. The structure for Qwen2's MLP block is:
*  A gate linear projection ($W_{gate}$).
*  An up linear projection ($W_{up}$).
*  A SiLU activation function applied to the output of $W_{gate}$.
*  An element-wise multiplication of the SiLU-activated $W_{gate}$ output and the $W_{up}$ output. This forms the "gated" part.
*  A final down linear projection ($W_{down}$).

This can be expressed as:
$$
\text{MLP}(x) = (\text{SiLU}(W_{gate}(x)) \odot W_{up}(x))W_{down}
$$
Where $\odot$ denotes element-wise multiplication. All linear projections in Qwen2's MLP are typically implemented without bias.

```
N.. is zero or more dimensions for batches
E is hidden_size (embedding dimension of the model)
I is intermediate_size (dimension of the hidden layer in MLP)
L is the sequence length

input: N.. x L x E
w_gate: I x E
w_up: I x E
w_down: E x I
output: N.. x L x E
```

You can test your implementation by running:

```bash
pdm run test --week 1 --day 4 -- -k task_2
```

At the end of the day, you should be able to pass all tests of this day:

```bash
pdm run test --week 1 --day 4
```


{{#include copyright.md}}