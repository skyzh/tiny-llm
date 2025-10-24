import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        *batch_dims, l, e = x.shape
        # (B, L, E) * (Hq*D, E)T -> (B, L, Hq*D) -> (B, L, Hq, D)
        q = linear(x, self.wq, self.bq).reshape(*batch_dims, l, self.num_heads, -1)

        # (B, L, E) * (H*D, E)T -> (B, L, H*D) -> (B, L, H, D)
        k = linear(x, self.wk, self.bk).reshape(*batch_dims, l, self.num_kv_heads, -1)
        v = linear(x, self.wv, self.bv).reshape(*batch_dims, l, self.num_kv_heads, -1)
        # d_dim is the dimension of each token's query/key/value representation
        d_dim = q.shape[-1]
        rope = RoPE(dims=d_dim, seq_len=self.max_seq_len, base=self.theta, traditional=False)
        # q dims: (B, L, Hq, D) before and after
        q = rope(q, offset=slice(0, l))
        # k dims: (B, L, H, D) before and after
        k = rope(k, offset=slice(0, l))

        # need to convert:
        # q: (B, L, Hq, D) - > (B, Hq, L, D)
        # k and v: (B, L, H, D) -> (B, H, L, D)
        # x: (B, Hq, L, D)
        x = scaled_dot_product_attention_grouped(query=q.swapaxes(-2, -3),
                                                 key=k.swapaxes(-2, -3),
                                                 value=v.swapaxes(-2, -3),
                                                 mask=mask)
        # x: (B, Hq, L, D) -> (B, L, Hq, D) -> (B, L, Hq*D)
        x = x.swapaxes(-2, -3).reshape(*batch_dims, l, -1)

        # wo: (Hq*D, E)
        # returns: (B, L, E)
        return linear(x, self.wo)



class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        # w_gate: (I, E)
        # x: (N.., L, E)
        # wgatex: (N.., L, I)
        wgatex = linear(x, self.w_gate)
        # w_up: (I, E)
        # w_up: (N.., L, I)
        wupx = linear(x, self.w_up)

        # (N.., L, I)
        silu_left_factor = silu(wgatex) * wupx

        # (N.., L, I) * (E, I)T -> (N.., L, E)
        return linear(silu_left_factor, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
