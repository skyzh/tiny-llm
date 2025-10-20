import mlx.core as mx

from .basics import linear, softmax


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        embed_dim = query.shape[-1]
        scale = mx.rsqrt(embed_dim)
    qk_t = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        qk_t += mask
    return mx.matmul(softmax(qk_t, axis=-1), value)


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
        self.E = hidden_size
        self.H = num_heads
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
        # subspace_q = mx.matmul(query, self.wq.T).reshape(self.H, -1)
        # subspace_k = mx.matmul(key, self.wk.T).reshape(self.H, -1)
        # subspace_v = mx.matmul(value, self.wv.T).reshape(self.H, -1)

        ss_q = mx.matmul(query, self.wq.T)
        ss_q = ss_q.reshape(*ss_q.shape[:-1], self.H, -1).swapaxes(-2, -3)

        ss_k = mx.matmul(key, self.wk.T)
        ss_k = ss_k.reshape(*ss_k.shape[:-1], self.H, -1).swapaxes(-2, -3)

        # dims: N x L x (H x D)
        ss_v = mx.matmul(value, self.wv.T)
        # dims: N x H x L x D
        ss_v = ss_v.reshape(*ss_v.shape[:-1], self.H, -1).swapaxes(-2, -3)

        # dims: N x H x L x D
        sdpa = scaled_dot_product_attention_simple(ss_q, ss_k, ss_v, mask=mask)

        # N x H x L x D -> N x L x H x D -> N x L x (H x D)
        sdpa = sdpa.swapaxes(-2, -3)
        sdpa = sdpa.reshape(*sdpa.shape[:-2], -1)

        return linear(sdpa, self.wo)


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
    mask: mx.array | None = None,
) -> mx.array:
    pass
    pass
    pass
