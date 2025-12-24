import torch
import torch.nn.functional as F
from typing import Optional


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_max = x.max(dim=axis, keepdim=True).values
    e_x = torch.exp(x - x_max)
    result = e_x / e_x.sum(dim=axis, keepdim=True)
    return result.to(orig_dtype)




def linear(x, w, bias=None):
    if torch.isnan(x).any():
        raise ValueError("NaN detected in input x")
    if torch.isnan(w).any():
        raise ValueError("NaN detected in weights w")
    if bias is not None and torch.isnan(bias).any():
        raise ValueError("NaN detected in bias")
    return F.linear(x, w, bias)



def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    return F.linear(x, w, bias)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)