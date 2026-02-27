import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, weight: torch.Tensor | None = None, eps: float = 1e-6):
        super().__init__()
        if weight is None:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = nn.Parameter(weight)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.to(torch.float32)
        norm = torch.mean(x_float ** 2, dim=-1, keepdim=True)
        norm = torch.rsqrt(norm + self.eps)
        output = (x * norm.to(x.dtype)) * self.weight
        return output


