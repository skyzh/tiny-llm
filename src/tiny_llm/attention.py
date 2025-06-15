import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value)


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
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        batch_size = query.shape[0]
        # query, key, value: N x L x E

        # 0. Compute linear projections
        query = linear(query, self.wq)
        key = linear(key, self.wk)
        value = linear(value, self.wv)

        # 1. Reshape to (batch_size, num_heads, seq_len, hidden_size)
        query = mx.reshape(
            query, (batch_size, -1, self.num_heads, self.hidden_size // self.num_heads))
        key = mx.reshape(
            key, (batch_size, -1, self.num_heads, self.hidden_size // self.num_heads))
        value = mx.reshape(
            value, (batch_size, -1, self.num_heads, self.hidden_size // self.num_heads))

        # 2. Transpose to (batch_size, num_heads, seq_len, hidden_size)
        query = mx.transpose(query, axes=(0, 2, 1, 3))
        key = mx.transpose(key, axes=(0, 2, 1, 3))
        value = mx.transpose(value, axes=(0, 2, 1, 3))

        # 3. Compute scaled dot product attention
        output = scaled_dot_product_attention_simple(
            query, key, value, mask=mask)

        # 4. concat heads
        output = mx.reshape(mx.transpose(output, axes=(
            0, 2, 1, 3)), (batch_size, -1, self.hidden_size))

        # 5. Compute linear projection
        output = linear(output, self.wo)

        return output


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
