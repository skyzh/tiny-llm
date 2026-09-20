# ruff: noqa: F401, F811

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mlx.core as mx

from .embedding import Embedding
from .kv_cache import TinyKvCache
from .quantize import QuantizedWeights, dequantize_linear
from .week2_kernels import (
    FastRMSNorm,
    FastRoPE,
    io_aware_dense_attention,
    quantized_gate_up_swiglu,
    quantized_qkv,
    scaled_dot_product_attention,
    supports_fused_gate_up,
    supports_shared_input_qkv,
    swiglu,
)


@dataclass(frozen=True)
class Week2CheckpointFeatures:
    quantized_weights: bool = False
    fast_rms_norm: bool = False
    fast_rope: bool = False
    fast_swiglu: bool = False
    simdgroup_matmul: bool = False
    shared_input_qkv: bool = False
    shared_input_gate_up_swiglu: bool = False
    io_aware_dense_attention: bool = False


WEEK2_CHECKPOINT_FEATURES = MappingProxyType(
    {
        "kv-cache": Week2CheckpointFeatures(),
        "quantized-matvec": Week2CheckpointFeatures(quantized_weights=True),
        "simd-matmul": Week2CheckpointFeatures(
            quantized_weights=True,
            simdgroup_matmul=True,
        ),
        "rmsnorm": Week2CheckpointFeatures(
            quantized_weights=True,
            fast_rms_norm=True,
            simdgroup_matmul=True,
        ),
        "rope": Week2CheckpointFeatures(
            quantized_weights=True,
            fast_rms_norm=True,
            fast_rope=True,
            simdgroup_matmul=True,
        ),
        "swiglu": Week2CheckpointFeatures(
            quantized_weights=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            simdgroup_matmul=True,
        ),
        "shared-input-qkv": Week2CheckpointFeatures(
            quantized_weights=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            simdgroup_matmul=True,
            shared_input_qkv=True,
        ),
        "shared-input-gate-up-swiglu": Week2CheckpointFeatures(
            quantized_weights=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            simdgroup_matmul=True,
            shared_input_qkv=True,
            shared_input_gate_up_swiglu=True,
        ),
        "io-aware-dense-attention": Week2CheckpointFeatures(
            quantized_weights=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            simdgroup_matmul=True,
            shared_input_qkv=True,
            shared_input_gate_up_swiglu=True,
            io_aware_dense_attention=True,
        ),
    }
)
WEEK2_CHECKPOINTS = tuple(WEEK2_CHECKPOINT_FEATURES)

LEGACY_CHECKPOINT_REPLACEMENTS = MappingProxyType(
    {
        "decode-attention": "io-aware-dense-attention",
        "long-context-attention": "io-aware-dense-attention",
        "context-selected-attention": "io-aware-dense-attention",
        "split-k": "shared-input-gate-up-swiglu",
        "fused-gate-up": "shared-input-gate-up-swiglu",
        "prefill-fused-gate-up": "shared-input-gate-up-swiglu",
    }
)


def should_use_io_aware_dense_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: mx.array | str | None,
    *,
    enabled: bool,
) -> bool:
    pass


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_shared_input_qkv: bool = False,
        use_io_aware_dense_attention: bool = False,
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
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        use_fast_swiglu: bool = True,
        use_shared_input_gate_up_swiglu: bool = False,
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
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_fast_swiglu: bool = True,
        use_shared_input_qkv: bool = False,
        use_shared_input_gate_up_swiglu: bool = False,
        use_io_aware_dense_attention: bool = False,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen3ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        checkpoint: str = "io-aware-dense-attention",
        use_mlx_quantized_linear: bool = False,
        disable_shared_input_qkv: bool = False,
        disable_shared_input_gate_up_swiglu: bool = False,
        disable_io_aware_dense_attention: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        pass

    def create_kv_cache(self) -> list[TinyKvCache]:
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        pass
