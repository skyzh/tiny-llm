import mlx.core as mx
from .basics import softmax, linear

def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    L, D = query.shape[-2], query.shape[-1]

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
        query = query.swapaxes(-1, -2)
        key = key.swapaxes(-1, -2)
        value = value.swapaxes(-1, -2)
        
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
    mask = mx.tril(mx.ones((L, S)), k=(S-L))
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask

def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    H_q, L, D = query.shape[-3], query.shape[-2], query.shape[-1]
    H, S = key.shape[-3], key.shape[-2]

    assert H_q % H == 0, "Query heads must be divisible by kv heads"

    n_repeats = H_q // H
    
    q_shape = query.shape
    query = query.reshape(-1, H, n_repeats, L, D)
    key = key.reshape(-1, H, 1, S, D)
    value = value.reshape(-1, H, 1, S, D)

    print("query shape: ", query.shape)
    print("key shape: ", key.shape)
    print("value shape: ", value.shape) 

    score = mx.matmul(query, key.swapaxes(-2, -1))
    atten_score = score * (mx.rsqrt(D) if scale is None else scale)

    if isinstance(mask, str) and mask == "causal":
        mask = causal_mask(L, S, atten_score.dtype)
        atten_score += mask
    elif isinstance(mask, mx.array):
        atten_score += mask.reshape(atten_score.shape)
    elif mask is None:
        pass
    else:
        raise NotImplementedError
    
    return mx.matmul(softmax(atten_score, axis=-1), value).reshape(q_shape)
    


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
