import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    attention_weights = softmax(scores, axis=-1)

    return mx.matmul(attention_weights, value)


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
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
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
        query = linear(query, self.wq)
        key = linear(key, self.wk)
        value = linear(value, self.wv)

        query = query.reshape(query.shape[:-1] + (self.num_heads, self.head_size))
        key = key.reshape(key.shape[:-1] + (self.num_heads, self.head_size))
        value = value.reshape(value.shape[:-1] + (self.num_heads, self.head_size))

        query = query.swapaxes(-3, -2)
        key = key.swapaxes(-3, -2)
        value = value.swapaxes(-3, -2)

        output = scaled_dot_product_attention_simple(query, key, value, mask=mask)
        output = output.swapaxes(-3, -2)
        output = output.reshape(output.shape[:-2] + (self.hidden_size,))
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
