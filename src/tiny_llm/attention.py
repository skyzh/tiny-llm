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
        return torch.matmul(softmax(scaled_qk + mask, axis=-1), value)
    else:
        return torch.matmul(softmax(scaled_qk, axis=-1), value)


class SimpleMultiHeadAttention:
    #    Input (N, L, E)
    #        │
    #        ├── linear(·, wq) → (N, L, H*D) → reshape → (N, L, H, D) → transpose → (N, H, L, D) ─┐
    #        ├── linear(·, wk) → (N, L, H*D) → reshape → (N, L, H, D) → transpose → (N, H, L, D) ─┤ attention → (N, H, L, D)
    #        └── linear(·, wv) → (N, L, H*D) → reshape → (N, L, H, D) → transpose → (N, H, L, D) ─┘
    #                                                                                                  │
    #                                                                                                  ▼
    #                                                                        transpose → (N, L, H, D) → reshape → (N, L, H*D)
    #                                                                                                  │
    #                                                                                        linear(·, wo)
    #                                                                                                  │
    #                                                                                                  ▼
    #                                                                                          Output (N, L, E)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        self.head_dim = self.hidden_size // self.num_heads

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._linear_transpose(query, self.wq)
        k = self._linear_transpose(key, self.wk)
        v = self._linear_transpose(value, self.wv)
        out = scaled_dot_product_attention_simple(
            query=q,
            key=k,
            value=v,
            mask=mask,
        )
        out = out.transpose(-3, -2)
        out = out.reshape(*out.shape[:-2], self.num_heads * self.head_dim)

        return linear(out, self.wo)

    def _linear_transpose(self, x: torch.Tensor, wX: torch.Tensor) -> torch.Tensor:
        X = linear(x, wX)
        X = X.reshape(
            *X.shape[:-1],
            self.num_heads,
            self.head_dim,
        )
        return X.transpose(-3, -2)


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
