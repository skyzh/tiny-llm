import torch
import torch.nn.functional as F
import math


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    # TODO: manual implementation
    return F.softmax(x, dim=axis)


def linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    pass


def silu(x: torch.Tensor) -> torch.Tensor:
    pass
