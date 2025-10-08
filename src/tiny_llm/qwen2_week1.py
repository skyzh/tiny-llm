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
        self.d_k = hidden_size // num_heads
        self.rope = RoPE(self.d_k, max_seq_len, theta, False)   # use untroditional RoPE
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 1. liear
        N, L, _ = x.shape   # x: (N, L, d_model)
        query = (
            linear(x, self.wq, self.bq)  # query: (N, L, d_model)
            .reshape(N, L, self.num_heads, self.d_k)    # query: (N, L, h, d_k)
            # .transpose(0, 2, 1, 3)  # query: (N, h, L, d_k)
        )
        key = (
            linear(x, self.wk, self.bk)
            .reshape(N, L, self.num_kv_heads, self.d_k)
        )
        value = (
            linear(x, self.wv, self.bv)
            .reshape(N, L, self.num_kv_heads, self.d_k)
        )

        # 2. RoPE
        query = self.rope(query, offset=slice(0, L))
        key = self.rope(key, offset=slice(0, L))
        query = query.transpose(0, 2, 1, 3)  # query: (N, h, L, d_k)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        # 2-3. scaled dot-product attention & concat
        scale = 1.0 / mx.sqrt(self.d_k)   # scale = 1 / sqrt(d_k)
        scores = (
            scaled_dot_product_attention_grouped(query, key, value, scale, mask=mask)  # scores: (N, h, L, d_k)
            .transpose(0, 2, 1, 3)  # scores: (N, L, h, d_k)
            .reshape(N, L, self.hidden_size)  # scores: (N, L, d_model)
        )

        # 4. linear
        return linear(scores, self.wo)  # output: (N, L, d_model)


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
        # mlp(x) = ((silu(x * w_gate.T) @ (x * w_up.T))) @ w_down.T
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


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
