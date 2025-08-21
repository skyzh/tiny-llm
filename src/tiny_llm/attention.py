import mlx.core as mx
from .basics import softmax, linear

def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    D, L = query.shape[-1], query.shape[-2]

    score = mx.matmul(query, key.swapaxes(-2, -1))
    atten_score = score * (mx.rsqrt(D) if scale is None else scale)

    if mask is not None:
        atten_score += mask

    return mx.matmul(softmax(atten_score, axis=-1), value)

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
        self.head_hidden_size = self.hidden_size // self.num_heads

        # E x (H x D)
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
        N, L = query.shape[0], query.shape[1]

        # N x L x H x D
        query = linear(query, self.wq).reshape(N, L, self.num_heads, self.head_hidden_size)
        key = linear(key, self.wk).reshape(N, L, self.num_heads, self.head_hidden_size)
        value = linear(value, self.wv).reshape(N, L, self.num_heads, self.head_hidden_size)

        # N x H x L x D
        query = query.swapaxes(1, 2)
        key = key.swapaxes(1, 2)
        value = value.swapaxes(1, 2)
        
        # N x H x L x D
        mh_attention = scaled_dot_product_attention_simple(
                       query,
                       key,
                       value,
                       mask=mask)
        # N x L x H x D
        mh_attention = mh_attention.swapaxes(1, 2)
        
        # N x L x E
        mh_attention = mh_attention.reshape(N, L, -1)
        
        out = linear(mh_attention, self.wo)
        return out
        


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
