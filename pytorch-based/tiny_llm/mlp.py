import torch
import torch.nn as nn
from .basics import silu


class Qwen2MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor):
        super().__init__()

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        self.gate_proj.weight.data.copy_(w_gate.to(torch.float32))
        self.up_proj.weight.data.copy_(w_up.to(torch.float32))
        self.down_proj.weight.data.copy_(w_down.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        self.gate_proj.weight.data = self.gate_proj.weight.data.to(dtype)
        self.up_proj.weight.data = self.up_proj.weight.data.to(dtype)
        self.down_proj.weight.data = self.down_proj.weight.data.to(dtype)

        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        gated = silu(gate_out) * up_out
        out = self.down_proj(gated)
        return out
