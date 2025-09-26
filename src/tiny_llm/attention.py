import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    scale_factor = scale if scale is not None else 1.0 / mx.sqrt(query.shape[-1])   # scale_factor = 1 / sqrt(d_k)
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * scale_factor # scores = q @ k^T / sqrt(d_k)
    if mask is not None:
        scores  = scores + mask
    scores = mx.matmul(softmax(scores, axis=-1), value)   # scores = softmax(scores) @ v
    return scores


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
        self.d_model = hidden_size
        self.h = num_heads
        assert hidden_size % num_heads == 0
        self.d_k = hidden_size // num_heads
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
        # 1. liear
        N, L, _ = query.shape   # query: (N, L, d_model)
        query = (
            linear(query, self.wq)  # query: (N, L, d_model)
            .reshape(N, L, self.h, self.d_k)    # query: (N, L, h, d_k)
            .transpose(0, 2, 1, 3)  # query: (N, h, L, d_k)
        )
        key = (
            linear(key, self.wk)
            .reshape(N, L, self.h, self.d_k)
            .transpose(0, 2, 1, 3)
        )
        value = (
            linear(value, self.wv)
            .reshape(N, L, self.h, self.d_k)
            .transpose(0, 2, 1, 3)
        )

        # 2-3. scaled dot-product attention & concat
        scale = 1.0 / mx.sqrt(self.d_k)   # scale = 1 / sqrt(d_k)
        scores = (
            scaled_dot_product_attention_simple(query, key, value, scale, mask=mask)  # scores: (N, h, L, d_k)
            .transpose(0, 2, 1, 3)  # scores: (N, L, h, d_k)
            .reshape(N, L, self.d_model)  # scores: (N, L, d_model)
        )

        # 4. linear
        return linear(scores, self.wo)  # output: (N, L, d_model)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L, S)), k=(S - L))
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask

def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    original_shape = query.shape
    h_q, L, d_k = query.shape[-3:]
    h_k, S, _ = key.shape[-3:]
    n_requests = h_q // h_k

    N = query.shape[:-3]
    query.reshape(*N, -1, n_requests, h_k, L, d_k)
    key.reshape(*N, -1, 1, h_k, S, d_k)
    value.reshape(*N, -1, 1, h_k, S, d_k)

    scale_factor = scale if scale is not None else 1.0 / mx.sqrt(d_k)   # scale = 1 / sqrt(d_k)
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * scale_factor # scores : (*N, 1, n_requests, h_k, L, S)
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, query.dtype)   # mask : (L, S)
        else:
            mask = mx.broadcast_to(mask, (*N, h_q, L, S))  # mask : (*N, h_q, L, S)
            mask = mask.reshape(*N, 1, h_k, n_requests, L, S)  # mask : (*N, 1, n_requests, h_k, L, S)
        scores = scores + mask
    scores = mx.matmul(softmax(scores, axis=-1), value)   # scores : (*N, 1, n_requests, h_k, L, d_k)
    return scores.reshape(original_shape)   # scores : (*N, h_q, L, d_k)



def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
