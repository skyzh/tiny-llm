import mlx.core as mx
from .basics import softmax, linear
import math

def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    scale_factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    atten_score = query @ key.swapaxes(-2, -1) * scale_factor
    if mask is not None:
        atten_score += mask
    return mx.softmax(atten_score, -1) @ value


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        assert wq.shape == (num_heads * self.head_dim, hidden_size)
        assert wk.shape == (num_heads * self.head_dim, hidden_size)
        assert wv.shape == (num_heads * self.head_dim, hidden_size)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        q_proj = linear(query, self.wq).reshape(N, L, self.num_heads, self.head_dim).swapaxes(-3, -2)
        k_proj = linear(key, self.wk).reshape(N, L, self.num_heads, self.head_dim).swapaxes(-3, -2)
        v_proj = linear(value, self.wv).reshape(N, L, self.num_heads, self.head_dim).swapaxes(-3, -2)
        
        atten = scaled_dot_product_attention_simple(q_proj, k_proj, v_proj, self.scale, mask)
        output = linear(atten.swapaxes(-3, -2).reshape(N, L, self.hidden_size), self.wo)
        return output

def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L, S)), k=S-L)
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf))
    return mask.astype(dtype)

def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    factor = mx.array(factor, dtype=query.dtype)

    B = query.shape[:-3]
    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    assert H_q % H == 0
    n_repeats = H_q // H

    query = query.reshape(*B, -1, H, n_repeats, L, D)
    key = key.reshape(*B, -1, H, 1, S, D).swapaxes(-2, -1)
    value = value.reshape(*B, -1, H, 1, S, D)

    score = mx.matmul(query, key) * factor

    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, score.dtype)
        else:
            mask = mx.broadcast_to(mask, (*B, H_q, L, S)).reshape(*B, -1, H, n_repeats, L, S)
        score += mask
    return mx.matmul(softmax(score, axis=-1), value).reshape(*B, H_q, L, D)

def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass
