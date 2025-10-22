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
    *batch_dims, h_q, l, d = query.shape
    *ignore, h, s, _ = key.shape
    assert not h_q % h, f"h should divide h_q but h={h} and h_q={h_q}"
    assert l <= s, "l should be no greater than s but l={l} and s={s}"
    n_repeat = h_q//h

    # truncate key and value to actual seq length of query
    # and add a dummy dimension for easy broadcasting later
    # (N.., H, S, D) -> (N.., H, 1, S, D)
    key_reshaped = key.reshape(*batch_dims, h, 1, s, d)
    value_reshaped = value.reshape(*batch_dims, h, 1, s, d)

    # reshape query tensor
    # (N.., H_q, L, D) -> (N.., H, n_repeat, L, D)
    query_reshaped = query.reshape(*batch_dims, h, n_repeat, l, d)

    if not scale:
        scale = mx.rsqrt(d)
    # qk_t shape: (N.., H, n_repeat, L, D) * (N.., H, 1, D, S) -> (N.., H, n_repeat, L, S)
    qk_t = mx.matmul(query_reshaped, key_reshaped.swapaxes(-2, -1)) * scale
    if isinstance(mask, mx.array):
        # mask (N.., H_q, L, S) -> (N.., H, n_repeat, L, S)
        qk_t += mask.reshape(*batch_dims, h, n_repeat, l, s)

    # shape: (N.., H, n_repeat, L, S)
    p_attn = softmax(qk_t, axis=-1)
    # shape: (N.., H, n_repeat, L, S) * (N.., H, 1, S, D) -> (N.., H, n_repeats, L, D) -> (N.., H_q, L, D)
    return mx.matmul(p_attn, value_reshaped).reshape(*batch_dims, h_q, l, d)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
