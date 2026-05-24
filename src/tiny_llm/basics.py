import torch
import torch.nn.functional as F
import math


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    # softmax(x_i) = e^(x_i) / Σ e^(x_j)
    x_shifted = x - x.max(dim=axis, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    sum_exp_x = exp_x.sum(dim=axis, keepdim=True)
    return exp_x / sum_exp_x


def linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # y = xW^T + b
    x_WT = x @ w.T  # torch.matmul(x, w.T)
    if bias is not None:
        return x_WT + bias
    else:
        return x_WT


def silu(x: torch.Tensor) -> torch.Tensor:
    pass
