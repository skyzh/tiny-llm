from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mlx.core as mx

from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .embedding import Embedding, QuantizedEmbedding
from .kv_cache import TinyKvCache
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .quantize import QuantizedWeights, dequantize_linear, quantized_linear
from .week2_kernels import (
    FastRMSNorm,
    FastRoPE,
    io_aware_dense_attention,
    quantized_qkv,
    quantized_gate_up_swiglu,
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


def _validate_checkpoint(checkpoint: str) -> None:
    replacement = LEGACY_CHECKPOINT_REPLACEMENTS.get(checkpoint)
    if replacement is not None:
        raise ValueError(
            f"Week 2 checkpoint {checkpoint!r} was replaced by {replacement!r}"
        )
    if checkpoint not in WEEK2_CHECKPOINTS:
        raise ValueError(
            f"unknown Week 2 checkpoint {checkpoint!r}; choose one of "
            f"{WEEK2_CHECKPOINTS}"
        )


def should_use_io_aware_dense_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: mx.array | str | None,
    *,
    enabled: bool,
) -> bool:
    if not enabled:
        return False
    if isinstance(mask, str) and mask != "causal":
        return False
    return (
        query.dtype in (mx.float32, mx.float16, mx.bfloat16)
        and query.dtype == key.dtype
        and query.dtype == value.dtype
        and query.ndim == 4
        and key.ndim == 4
        and key.shape == value.shape
        and query.shape[0] == key.shape[0]
        and query.shape[1] % key.shape[1] == 0
        and 0 < query.shape[-1] <= 256
        and query.shape[-1] == key.shape[-1]
        and 0 < query.shape[-2] <= key.shape[-2]
    )


def _linear(x: mx.array, weight: mx.array | QuantizedWeights) -> mx.array:
    if isinstance(weight, QuantizedWeights):
        return quantized_linear(x, weight)
    return linear(x, weight)


def _readable_rope_offset(
    offset: int | list[int] | mx.array, sequence_length: int
) -> slice | list[slice]:
    if isinstance(offset, int):
        return slice(offset, offset + sequence_length)
    if isinstance(offset, list):
        return [slice(value, value + sequence_length) for value in offset]
    values = offset.tolist()
    if not isinstance(values, list):
        values = [values]
    return [slice(value, value + sequence_length) for value in values]


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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.use_fast_rope = use_fast_rope
        self.use_shared_input_qkv = use_shared_input_qkv
        self.use_io_aware_dense_attention = use_io_aware_dense_attention
        # Earlier checkpoint tests use this internal flag only to prove that
        # dense attention is not enabled before its checkpoint.
        self.use_decode_attention = use_io_aware_dense_attention
        self.shared_input_qkv_dispatches = 0
        self.separate_qkv_dispatches = 0
        self.io_aware_dense_attention_dispatches = 0
        self.readable_attention_dispatches = 0
        rope_cls = FastRoPE if use_fast_rope else RoPE
        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        self.rope = rope_cls(self.head_dim, max_seq_len, theta)
        self.q_norm = norm_cls(self.head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = norm_cls(self.head_dim, k_norm, eps=rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        if self.use_shared_input_qkv and supports_shared_input_qkv(
            x, self.wq, self.wk, self.wv
        ):
            self.shared_input_qkv_dispatches += 1
            projection_q, projection_k, projection_v = quantized_qkv(
                x, self.wq, self.wk, self.wv
            )
        else:
            self.separate_qkv_dispatches += 1
            projection_q = _linear(x, self.wq)
            projection_k = _linear(x, self.wk)
            projection_v = _linear(x, self.wv)
        projection_q = projection_q.reshape(B, L, self.num_heads, self.head_dim)
        projection_k = projection_k.reshape(B, L, self.num_kv_heads, self.head_dim)
        projection_q = self.q_norm(projection_q)
        projection_k = self.k_norm(projection_k)
        projection_v = projection_v.reshape(B, L, self.num_kv_heads, self.head_dim)
        rope_offsets = offsets
        if not self.use_fast_rope:
            rope_offsets = _readable_rope_offset(rope_offsets, L)
        projection_q = self.rope(projection_q, offset=rope_offsets)
        projection_k = self.rope(projection_k, offset=rope_offsets)
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)
        projection_k, projection_v, _, mask = cache.update_and_fetch(
            projection_k, projection_v, mask_length=L, mask=mask
        )
        if should_use_io_aware_dense_attention(
            projection_q,
            projection_k,
            projection_v,
            mask,
            enabled=self.use_io_aware_dense_attention,
        ):
            self.io_aware_dense_attention_dispatches += 1
            x = io_aware_dense_attention(
                projection_q,
                projection_k,
                projection_v,
                scale=self.scale,
                mask=mask,
            )
        else:
            self.readable_attention_dispatches += 1
            x = scaled_dot_product_attention_grouped(
                projection_q.astype(mx.float32),
                projection_k.astype(mx.float32),
                projection_v.astype(mx.float32),
                scale=self.scale,
                mask=mask,
            ).astype(x.dtype)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return _linear(x, self.wo)


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
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.use_fast_swiglu = use_fast_swiglu
        self.use_shared_input_gate_up_swiglu = use_shared_input_gate_up_swiglu
        self.shared_input_gate_up_swiglu_dispatches = 0
        self.separate_gate_up_dispatches = 0

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_shared_input_gate_up_swiglu and supports_fused_gate_up(
            x, self.w_gate, self.w_up
        ):
            self.shared_input_gate_up_swiglu_dispatches += 1
            hidden = quantized_gate_up_swiglu(x, self.w_gate, self.w_up)
        else:
            self.separate_gate_up_dispatches += 1
            gate = _linear(x, self.w_gate)
            up = _linear(x, self.w_up)
            hidden = swiglu(gate, up) if self.use_fast_swiglu else silu(gate) * up
        return _linear(hidden, self.w_down)


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
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.mlp = Qwen3MLP(
            hidden_size,
            intermediate_size,
            w_gate,
            w_up,
            w_down,
            use_fast_swiglu=use_fast_swiglu,
            use_shared_input_gate_up_swiglu=use_shared_input_gate_up_swiglu,
        )
        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        self.input_layernorm = norm_cls(
            hidden_size, w_input_layernorm, eps=rms_norm_eps
        )
        self.post_attention_layernorm = norm_cls(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        self.self_attn = Qwen3MultiHeadAttention(
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
            use_fast_rms_norm=use_fast_rms_norm,
            use_fast_rope=use_fast_rope,
            use_shared_input_qkv=use_shared_input_qkv,
            use_io_aware_dense_attention=use_io_aware_dense_attention,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), offset, cache, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


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
        _validate_checkpoint(checkpoint)
        self.checkpoint = checkpoint
        features = WEEK2_CHECKPOINT_FEATURES[checkpoint]
        use_quantized_weights = features.quantized_weights
        use_fast_rms_norm = features.fast_rms_norm
        use_fast_rope = features.fast_rope
        use_fast_swiglu = features.fast_swiglu
        use_shared_input_qkv = (
            features.shared_input_qkv and not disable_shared_input_qkv
        )
        use_shared_input_gate_up_swiglu = (
            features.shared_input_gate_up_swiglu
            and not disable_shared_input_gate_up_swiglu
        )
        use_io_aware_dense_attention = (
            features.io_aware_dense_attention and not disable_io_aware_dense_attention
        )
        use_simdgroup_matmul = features.simdgroup_matmul
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.use_fast_rope = use_fast_rope
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        precision = mx.bfloat16
        self.precision = precision

        def model_weight(layer: Any) -> mx.array | QuantizedWeights:
            if use_quantized_weights:
                return QuantizedWeights.from_mlx_layer(
                    layer,
                    use_simdgroup_matmul=use_simdgroup_matmul,
                    use_mlx_quantized_linear=use_mlx_quantized_linear,
                )
            return dequantize_linear(layer).astype(mx.bfloat16)

        embedding_weight = model_weight(mlx_model.model.embed_tokens)
        if isinstance(embedding_weight, QuantizedWeights):
            self.embedding = QuantizedEmbedding(
                vocab_size=self.vocab_size,
                embedding_dim=self.hidden_size,
                weight=embedding_weight,
            )
        else:
            self.embedding = Embedding(
                vocab_size=self.vocab_size,
                embedding_dim=self.hidden_size,
                weight=embedding_weight,
            )
        self.layers_inner = []

        for i in range(mlx_model.args.num_hidden_layers):
            wq = model_weight(mlx_model.model.layers[i].self_attn.q_proj)
            wk = model_weight(mlx_model.model.layers[i].self_attn.k_proj)
            wv = model_weight(mlx_model.model.layers[i].self_attn.v_proj)
            wo = model_weight(mlx_model.model.layers[i].self_attn.o_proj)
            w_gate = model_weight(mlx_model.model.layers[i].mlp.gate_proj)
            w_up = model_weight(mlx_model.model.layers[i].mlp.up_proj)
            w_down = model_weight(mlx_model.model.layers[i].mlp.down_proj)

            layer = Qwen3TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                head_dim=mlx_model.args.head_dim,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                q_norm=mlx_model.model.layers[i].self_attn.q_norm.weight,
                k_norm=mlx_model.model.layers[i].self_attn.k_norm.weight,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight,
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
                use_fast_rms_norm=use_fast_rms_norm,
                use_fast_rope=use_fast_rope,
                use_fast_swiglu=use_fast_swiglu,
                use_shared_input_qkv=use_shared_input_qkv,
                use_shared_input_gate_up_swiglu=use_shared_input_gate_up_swiglu,
                use_io_aware_dense_attention=use_io_aware_dense_attention,
            )
            self.layers_inner.append(layer)
        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        self.norm = norm_cls(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight,
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = model_weight(mlx_model.lm_head)
        else:
            self.w_lm_head = None
        self.mlx_model = mlx_model

    def dispatch_counters(self) -> dict[str, int]:
        return {
            "shared_input_qkv": sum(
                layer.self_attn.shared_input_qkv_dispatches
                for layer in self.layers_inner
            ),
            "separate_qkv": sum(
                layer.self_attn.separate_qkv_dispatches for layer in self.layers_inner
            ),
            "io_aware_dense_attention": sum(
                layer.self_attn.io_aware_dense_attention_dispatches
                for layer in self.layers_inner
            ),
            "readable_attention": sum(
                layer.self_attn.readable_attention_dispatches
                for layer in self.layers_inner
            ),
            "shared_input_gate_up_swiglu": sum(
                layer.mlp.shared_input_gate_up_swiglu_dispatches
                for layer in self.layers_inner
            ),
            "separate_gate_up": sum(
                layer.mlp.separate_gate_up_dispatches for layer in self.layers_inner
            ),
        }

    def create_kv_cache(self) -> list[TinyKvCache]:
        from .kv_cache import TinyKvFullCache

        return [TinyKvFullCache() for _ in range(self.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        if isinstance(offset, int):
            for layer, layer_cache in enumerate(cache):
                cache_offset = getattr(layer_cache, "offset", None)
                if cache_offset is not None and cache_offset != offset:
                    raise ValueError(
                        f"layer {layer} cache offset {cache_offset} "
                        f"does not match model offset {offset}"
                    )
        h = self.embedding(inputs)
        mask = None if inputs.shape[1] == 1 else "causal"
        if not getattr(self, "use_fast_rope", True):
            rope_offsets = offset
        elif isinstance(offset, int):
            rope_offsets = mx.full((inputs.shape[0],), offset, dtype=mx.int32)
        elif isinstance(offset, list):
            rope_offsets = mx.array(offset, dtype=mx.int32)
        else:
            rope_offsets = offset
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, rope_offsets, cache[layer], mask=mask)
        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            h = h[:, -logits_to_keep:, :]
        h = self.norm(h)
        if self.w_lm_head is not None:
            return _linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)
