import mlx.core as mx
from typing import Any
from .kv_cache import TinyKvCache
from .moe import Moe
from .quantize import QuantizedWeights


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_flash_attention: bool = False,
        use_paged_attention: bool = True,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        mlp: Qwen3MLP | Moe,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
        use_paged_attention: bool = True,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


def is_qwen3_moe_sparse_layer(args: Any, layer_idx: int) -> bool:
    pass


class Qwen3ModelWeek3:
    def __init__(
        self,
        mlx_model: Any,
        page_size: int = 128,
        enable_flash_attn: bool | None = None,
        enable_performance_lab: bool = False,
        enable_paged_attention: bool = True,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        pass

    def create_kv_cache(self) -> list[TinyKvCache]:
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int | list[int] | mx.array,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        pass
