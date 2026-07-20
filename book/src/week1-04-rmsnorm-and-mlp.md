# Week 1 Day 4: RMSNorm and the Multilayer Perceptron

On Day 4, we will implement two important components of the Qwen3 Transformer architecture: RMSNorm and the multilayer
perceptron (MLP), also known as the feed-forward network. RMSNorm is a normalization technique with less computational
overhead than traditional layer normalization. The MLP applies nonlinear transformations after the attention block.

## Task 1: Implement `RMSNorm`

In this task, we will implement the `RMSNorm` layer.

```
src/tiny_llm/layer_norm.py
```

Day 3 used `mx.fast.rms_norm` directly so that the GQA chapter could stay focused on attention. This task implements the
same normalization rule as a reusable layer. From this point on, the Transformer block, final model normalization, and
Q/K normalization path can use your `RMSNorm` implementation.

**📚 Readings**

* [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
* [Qwen3 layers implementation in mlx-lm (includes RMSNorm)](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py) - See `RMSNorm`.


RMSNorm is defined as:

$$
y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \text{weight}
$$

where:

- `x` is the input tensor.
- `weight` is a learned scaling parameter.
- `epsilon` (`eps`) is a small constant, such as `1e-5` or `1e-6`, added for numerical stability.
- `mean(x^2)` is the mean of the squared elements along the final dimension.

Apply normalization independently to each feature vector along the input's final dimension. Cast the input to `float32`
for the normalization calculation, including the mean, to preserve precision when the original values use `float16` or
`bfloat16`. Cast the normalized value back to the input dtype before applying `weight`. This matches the low-precision
path used by MLX's fast RMSNorm kernels: normalization statistics are accumulated in `float32`, while the final scaling
happens in the model dtype.

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

In this task, we will implement the MLP block named `Qwen3MLP`.

```
src/tiny_llm/qwen3_week1.py
```

The original Transformer uses a simple position-wise feed-forward network (FFN) in each block. It consists of two linear
transformations with a ReLU activation between them.

Modern Transformer architectures, including Qwen3, often use more advanced FFN variants. Qwen3 uses SwiGLU, a gated linear
unit (GLU) variant.

A plain FFN can be abstracted as:

```plain
h = activation(W_up(x))
out = W_down(h)
```

A GLU keeps the same expand-then-project-back shape but adds another projection that gates the intermediate features before
`W_down`. This gives the MLP a learned, input-dependent way to control which intermediate channels matter, rather than
applying an activation only to the features produced by `W_up`.

SwiGLU is the GLU variant used by Qwen3:

```plain
u = W_up(x)
g = SiLU(W_gate(x))
out = W_down(g * u)
```

**📚 Readings**

- [Attention is All You Need (Transformer Paper, Section 3.3 "Position-wise Feed-Forward Networks")](https://arxiv.org/abs/1706.03762)
- [GLU paper: Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083)
- [SiLU (Swish) activation function](https://arxiv.org/pdf/1710.05941)
- [SwiGLU paper: GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1)
- [PyTorch SiLU documentation](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
- [Qwen3 layers implementation in mlx-lm (includes MLP)](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)

SwiGLU combines a GLU with the SiLU (sigmoid linear unit) activation function:

- A GLU gates one linear projection of the input with another, using element-wise multiplication to control which features
  pass through.
- SiLU is a smooth, non-monotonic activation function. Unlike ReLU, it has no zero-gradient region across all negative
  inputs and can produce nonzero outputs for negative values.

First, implement `silu` in `basics.py`. It takes a tensor of shape `N.. x I` and returns a tensor with the same shape:

$$
\text{SiLU}(x) = x * \text{sigmoid}(x) = \frac{x}{1 + e^{-x}}
$$
Compute the sigmoid part in a numerically stable way:

```text
if x >= 0:
    sigmoid(x) = 1 / (1 + exp(-x))
else:
    sigmoid(x) = exp(x) / (1 + exp(x))
```

The negative branch is algebraically equivalent to the direct sigmoid formula, but it prevents `exp(-x)` from becoming
`exp(large positive)` when `x` is a large negative value. In vector code, compute the direct branch with `abs(x)`, then
use `1 - y` for negative inputs. This approach closely matches MLX's low-precision GPU path.

Then implement `Qwen3MLP`. Qwen3's MLP contains:

- A gate projection ($W_{gate}$)
- An up projection ($W_{up}$)
- SiLU applied to the gate projection's output
- An element-wise product of the activated gate output and the up-projection output
- A final down projection ($W_{down}$)

This can be expressed as:

$$
\text{MLP}(x) = W_{down}(\text{SiLU}(W_{gate}(x)) \odot W_{up}(x))
$$
where $\odot$ denotes element-wise multiplication. Qwen3's MLP projections do not use biases.

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
