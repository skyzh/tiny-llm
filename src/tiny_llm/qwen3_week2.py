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
    scaled_dot_product_attention,
    swiglu,
)


@dataclass(frozen=True)
class Week2CheckpointFeatures:
    quantized_weights: bool = False
    fast_rms_norm: bool = False
    fast_rope: bool = False
    fast_swiglu: bool = False
    simdgroup_matmul: bool = False
    context_selected_attention: bool = False
    prefill_fused_gate_up: bool = False


WEEK2_CHECKPOINT_FEATURES = MappingProxyType(
    {
        "kv-cache": Week2CheckpointFeatures(),
        "quantized-matvec": Week2CheckpointFeatures(quantized_weights=True),
        "rmsnorm": Week2CheckpointFeatures(quantized_weights=True, fast_rms_norm=True),
        "rope": Week2CheckpointFeatures(
            quantized_weights=True, fast_rms_norm=True, fast_rope=True
        ),
        "swiglu": Week2CheckpointFeatures(
            quantized_weights=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
        ),
        "simd-matmul": Week2CheckpointFeatures(
            quantized_weights=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            simdgroup_matmul=True,
        ),
        "context-selected-attention": Week2CheckpointFeatures(
            quantized_weights=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            simdgroup_matmul=True,
            context_selected_attention=True,
        ),
        "prefill-fused-gate-up": Week2CheckpointFeatures(
            quantized_weights=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            simdgroup_matmul=True,
            prefill_fused_gate_up=True,
        ),
    }
)
WEEK2_CHECKPOINTS = tuple(WEEK2_CHECKPOINT_FEATURES)

CONTEXT_SELECTED_ATTENTION_MIN_CONTEXT = 8192
CONTEXT_SELECTED_ATTENTION_MAX_CONTEXT = 32768
CONTEXT_SELECTED_ATTENTION_MAX_QUERY = 2
LONG_CONTEXT_ATTENTION_MAX_CONTEXT = CONTEXT_SELECTED_ATTENTION_MAX_CONTEXT
LONG_CONTEXT_ATTENTION_MAX_QUERY = CONTEXT_SELECTED_ATTENTION_MAX_QUERY
DECODE_ATTENTION_MAX_CONTEXT = CONTEXT_SELECTED_ATTENTION_MAX_CONTEXT
DECODE_ATTENTION_MAX_QUERY = CONTEXT_SELECTED_ATTENTION_MAX_QUERY

LEGACY_CHECKPOINT_REPLACEMENTS = MappingProxyType(
    {
        "decode-attention": "context-selected-attention",
        "long-context-attention": "context-selected-attention",
        "split-k": "prefill-fused-gate-up",
        "fused-gate-up": "prefill-fused-gate-up",
    }
)


def should_use_context_selected_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: mx.array | str | None,
    *,
    enabled: bool,
) -> bool:
    pass


# Internal compatibility alias; the learner-owned policy is the
# context-selected function above.
should_use_long_context_attention = should_use_context_selected_attention


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
        use_context_selected_attention: bool = True,
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
        use_prefill_fused_gate_up: bool = False,
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
        use_context_selected_attention: bool = True,
        use_prefill_fused_gate_up: bool = False,
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
        checkpoint: str = "prefill-fused-gate-up",
        use_mlx_quantized_linear: bool = False,
        disable_context_selected_attention: bool = False,
        disable_prefill_fused_gate_up: bool = False,
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
