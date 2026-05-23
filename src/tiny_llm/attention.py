import math
import torch
import torch.nn.functional as F
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:

    # output = softmax(Q · K^T / sqrt(d) + M) · V
    qk = torch.matmul(query, key.transpose(-2, -1))
    d = query.shape[-1]
    if scale:
        scaled_qk = qk * scale
    else:
        scaled_qk = qk * 1 / math.sqrt(d)
    if mask is not None:
        return torch.matmul(F.softmax(scaled_qk + mask, dim=-1), value)
    else:
        return torch.matmul(F.softmax(scaled_qk, dim=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ):
        pass

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass


def causal_mask(L: int, S: int, dtype: torch.dtype) -> torch.Tensor:
    pass


def scaled_dot_product_attention_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    pass


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pass
