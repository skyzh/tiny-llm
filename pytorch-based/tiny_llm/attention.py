import torch
import torch.nn.functional as F
from .basics import softmax, linear

import torch

def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    orig_dtype = query.dtype

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)

    scores = torch.matmul(query, key.transpose(-2, -1))

    if scale is None:
        scale = 1.0 / (key.size(-1) ** 0.5)
    scores = scores * scale

    if mask is not None:
        mask = mask.to(dtype=torch.float32, device=scores.device)
        scores = scores+mask

    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, value)

    return out.to(orig_dtype)




def scaled_dot_product_attention_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return scaled_dot_product_attention_simple(query, key, value, scale, mask)

def causal_mask(L: int, S: int, dtype=torch.float32) -> torch.Tensor:
    mask = torch.zeros((L, S), dtype=dtype)
    for i in range(L-1):
        start_pos = S-L+i+1
        mask[i, start_pos:] = float('-inf')
    print("******")
    print(L,S)
    print(mask)
    print("*********")
    return mask


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ):
        assert num_query_heads % num_kv_heads == 0, "query_heads must be divisible by kv_heads"

        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_query_heads
        self.kv_repeat = num_query_heads // num_kv_heads

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo


    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    )-> torch.Tensor:
        batch_size, seq_len, _ = query.size()

        q = F.linear(query, self.wq)
        k = F.linear(key, self.wk)
        v = F.linear(value, self.wv)

        def reshape_q(x):
            return x.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        def reshape_kv(x):
            return x.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = reshape_q(q)
        k = reshape_kv(k)
        v = reshape_kv(v)

        k = k.unsqueeze(1).repeat(1, self.kv_repeat, 1, 1, 1).view(batch_size, self.num_query_heads, seq_len, self.head_dim)
        v = v.unsqueeze(1).repeat(1, self.kv_repeat, 1, 1, 1).view(batch_size, self.num_query_heads, seq_len, self.head_dim)

        scale = 1.0 / (self.head_dim ** 0.5)
        context = scaled_dot_product_attention_simple(q, k, v, scale, mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = F.linear(context, self.wo)

        return output
