import mlx.core as mx
from .basics import softmax, linear
from extensions import tiny_llm_ext


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d_k = query.shape[-1]
    factor = mx.rsqrt(d_k) if scale is None else scale
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    scores = softmax(scores, axis=-1)
    return mx.matmul(scores, value)  # output is (N.., L, D)


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
        assert query.shape == key.shape == value.shape

        querys = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        values = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        output = scaled_dot_product_attention_simple(
            querys,
            keys,
            values,
            scale=None,
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_size)
        return linear(output, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    """Generate a causal mask for attention mechanism.

    Args:
        L (int): Length of the query sequence.
        S (int): Length of the key/value sequence.
        dtype (mx.Dtype): Data type of the output mask.

    Returns:
        mx.array: A (L, S) shaped array where positions that should not be attended to are set to -inf,
                   and positions that can be attended to are set to 0.
    """
    mask = mx.full((L, S), -mx.inf, dtype=dtype)
    mask = mx.triu(mask, k=(S-L+1))
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    expected_shape = query.shape
    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    assert H_q % H == 0
    group = H_q // H

    query = query.reshape(-1, H, group, L, D)
    key = key.reshape(-1, H, 1, S, D)
    value = value.reshape(-1, H, 1, S, D)
    if mask == "causal":
        mask = causal_mask(L, S, query.dtype)
    elif mask is not None:
        mask = mask.reshape(-1, H, group, L, S)
    else:
        mask = None

    return scaled_dot_product_attention_simple(
        query,
        key,
        value,
        scale=scale,
        mask=mask,
    ).reshape(expected_shape)  # output is (N.., L, D)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    B, H_q, L, E = query.shape
    _, H, S, _ = key.shape
    assert H_q % H == 0

    query = query.reshape(-1, L, E)
    key = key.reshape(-1, S, E)
    value = value.reshape(-1, S, E)

    query = mx.contiguous(query)
    key = mx.contiguous(key)
    value = mx.contiguous(value)

    if mask is None:
        mask = mx.zeros((L, S))
    mask = mx.broadcast_to(mask, (B, H_q, L, S)).reshape(-1, L, S).astype(mx.float32)
    mask = mx.contiguous(mask)

    result = tiny_llm_ext.flash_attention(
        query,
        key,
        value,
        mask,
        scale,
        num_heads=H_q,
        num_kv_heads=H,
    )
    return mx.contiguous(result.reshape(B, H_q, L, E))
