from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mlx.core as mx
from extensions_ref.tiny_llm_ext_ref import _ext as tiny_llm_ext_ref_private

from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .embedding import Embedding, QuantizedEmbedding
from .kv_cache import TinyKvCache, TinyKvFullCache
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .quantize import QuantizedWeights, dequantize_linear, quantized_linear
from .week2_kernels import (
    DENSE_PREFILL_MIN_QUERY,
    _qk_norm_rope4,
    FastRMSNorm,
    FastRoPE,
    dense_prefill_attention_mma,
    decode_attention_online,
    swiglu,
)


@dataclass(frozen=True)
class Week2CheckpointFeatures:
    bounded_kv_capacity: bool = False
    quantized_weights: bool = False
    fast_rms_norm: bool = False
    fast_rope: bool = False
    fast_swiglu: bool = False
    simdgroup_matmul: bool = False
    tiled_prefill_attention: bool = False
    native_online_decode: bool = False
    fused_qk_norm_rope4: bool = False
    decode_submit18: bool = False


WEEK2_CHECKPOINT_FEATURES = MappingProxyType(
    {
        "kv-cache": Week2CheckpointFeatures(),
        "capacity-cache": Week2CheckpointFeatures(bounded_kv_capacity=True),
        "quantized-matvec": Week2CheckpointFeatures(
            bounded_kv_capacity=True, quantized_weights=True
        ),
        "simd-matmul": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
        ),
        "rmsnorm": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
        ),
        "rope": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
        ),
        "swiglu": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
        ),
        "tiled-prefill": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            tiled_prefill_attention=True,
        ),
        "selected": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            tiled_prefill_attention=True,
            native_online_decode=True,
            fused_qk_norm_rope4=True,
            decode_submit18=True,
        ),
    }
)
WEEK2_CHECKPOINTS = tuple(WEEK2_CHECKPOINT_FEATURES)


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
        use_tiled_prefill_attention: bool = False,
        use_native_online_decode: bool = False,
        use_fused_qk_norm_rope4: bool = False,
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
        self.use_tiled_prefill_attention = use_tiled_prefill_attention
        self.use_native_online_decode = (
            use_native_online_decode and use_tiled_prefill_attention
        )
        self.use_fused_qk_norm_rope4 = (
            use_fused_qk_norm_rope4 and use_fast_rms_norm and use_fast_rope
        )
        self.attention_dispatch_counts = {
            "tiled_prefill": 0,
            "tiled_prefill_fallback": 0,
            "readable": 0,
            "native_online_decode": 0,
            "native_split_decode": 0,
        }
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
        projection_q = _linear(x, self.wq).reshape(B, L, self.num_heads, self.head_dim)
        projection_k = _linear(x, self.wk).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        projection_q = self.q_norm(projection_q)
        projection_k = self.k_norm(projection_k)
        projection_v = _linear(x, self.wv).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        rope_offsets = offsets
        if not self.use_fast_rope:
            rope_offsets = _readable_rope_offset(rope_offsets, L)
        can_fuse_qk = (
            self.use_fused_qk_norm_rope4
            and type(self.q_norm) is FastRMSNorm
            and type(self.k_norm) is FastRMSNorm
            and type(self.rope) is FastRoPE
            and projection_q.shape == (1, 1, 32, 128)
            and projection_k.shape == (1, 1, 8, 128)
            and projection_q.dtype == mx.bfloat16
            and projection_k.dtype == mx.bfloat16
            and self.rope.dims == 128
            and not self.rope.traditional
        )
        if can_fuse_qk:
            projection_q = _qk_norm_rope4(projection_q, self.rope, rope_offsets)
            projection_k = _qk_norm_rope4(projection_k, self.rope, rope_offsets)
        else:
            projection_q = self.rope(projection_q, offset=rope_offsets)
            projection_k = self.rope(projection_k, offset=rope_offsets)
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)
        projection_k, projection_v, _, mask = cache.update_and_fetch(
            projection_k, projection_v, mask_length=L, mask=mask
        )
        can_tiled_prefill = (
            L >= DENSE_PREFILL_MIN_QUERY
            and projection_q.dtype == mx.bfloat16
            and self.head_dim == 128
        )
        if self.use_tiled_prefill_attention and not can_tiled_prefill:
            self.attention_dispatch_counts["tiled_prefill_fallback"] += 1

        if self.use_tiled_prefill_attention and can_tiled_prefill:
            self.attention_dispatch_counts["tiled_prefill"] += 1
            x = dense_prefill_attention_mma(
                projection_q,
                projection_k,
                projection_v,
                scale=self.scale,
                mask=mask,
            )
        elif (
            self.use_native_online_decode
            and self.use_tiled_prefill_attention
            and L == 1
            and self.head_dim == 128
            and 0 < projection_k.shape[2] <= 32768
            and all(
                a.dtype == mx.bfloat16
                for a in (projection_q, projection_k, projection_v)
            )
            and (mask is None or isinstance(mask, mx.array) or mask == "causal")
        ):
            self.attention_dispatch_counts["native_online_decode"] += 1
            if projection_k.shape[2] > 256:
                self.attention_dispatch_counts["native_split_decode"] += 1
            x = decode_attention_online(
                projection_q, projection_k, projection_v, self.scale, mask
            )
        else:
            self.attention_dispatch_counts["readable"] += 1
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
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.use_fast_swiglu = use_fast_swiglu

    def __call__(self, x: mx.array) -> mx.array:
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
        use_tiled_prefill_attention: bool = False,
        use_native_online_decode: bool = False,
        use_fused_qk_norm_rope4: bool = False,
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
            use_tiled_prefill_attention=use_tiled_prefill_attention,
            use_native_online_decode=use_native_online_decode,
            use_fused_qk_norm_rope4=use_fused_qk_norm_rope4,
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
    """Reference Week 2 model with an optimized ``selected`` default.

    ``tiled-prefill`` preserves the readable teaching decode path. The three
    reference scheduling controls default on only for ``selected`` and retain
    shape/dtype/cache guards. Pass False to disable a control explicitly; no
    experiment environment variables are read by this model.
    """

    def __init__(
        self,
        mlx_model: Any,
        checkpoint: str = "selected",
        use_mlx_quantized_linear: bool = False,
        use_bounded_kv_capacity: bool | None = None,
        use_register_cached_rms_norm: bool | None = None,
        use_tiled_prefill_attention: bool | None = None,
        use_native_online_decode: bool | None = None,
        use_fused_qk_norm_rope4: bool | None = None,
        use_decode_submit18: bool | None = None,
    ):
        if checkpoint not in WEEK2_CHECKPOINTS:
            raise ValueError(
                f"unknown Week 2 checkpoint {checkpoint!r}; "
                f"choose one of {WEEK2_CHECKPOINTS}"
            )
        for name, value in (
            ("use_bounded_kv_capacity", use_bounded_kv_capacity),
            ("use_register_cached_rms_norm", use_register_cached_rms_norm),
            ("use_tiled_prefill_attention", use_tiled_prefill_attention),
            ("use_native_online_decode", use_native_online_decode),
            ("use_fused_qk_norm_rope4", use_fused_qk_norm_rope4),
            ("use_decode_submit18", use_decode_submit18),
        ):
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool or None")
        self.checkpoint = checkpoint
        features = WEEK2_CHECKPOINT_FEATURES[checkpoint]
        if use_bounded_kv_capacity is None:
            use_bounded_kv_capacity = features.bounded_kv_capacity
        if use_register_cached_rms_norm is None:
            use_register_cached_rms_norm = features.fast_rms_norm
        if use_tiled_prefill_attention is None:
            use_tiled_prefill_attention = features.tiled_prefill_attention
        if use_native_online_decode is None:
            use_native_online_decode = features.native_online_decode
        if use_fused_qk_norm_rope4 is None:
            use_fused_qk_norm_rope4 = features.fused_qk_norm_rope4
        if use_decode_submit18 is None:
            use_decode_submit18 = features.decode_submit18
        use_quantized_weights = features.quantized_weights
        use_fast_rms_norm = use_register_cached_rms_norm
        use_fast_rope = features.fast_rope
        use_fast_swiglu = features.fast_swiglu
        use_simdgroup_matmul = features.simdgroup_matmul
        use_split_k_matmul = False
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.use_decode_submit18 = use_decode_submit18
        self.use_bounded_kv_capacity = use_bounded_kv_capacity
        self.use_register_cached_rms_norm = use_register_cached_rms_norm
        self.use_tiled_prefill_attention = use_tiled_prefill_attention
        self.mechanism_controls = MappingProxyType(
            {
                "capacity_cache": use_bounded_kv_capacity,
                "register_cached_rmsnorm": use_register_cached_rms_norm,
                "tiled_prefill": use_tiled_prefill_attention,
                "native_online_decode": use_native_online_decode,
                "fused_qk_norm_rope4": use_fused_qk_norm_rope4,
                "decode_submit18": use_decode_submit18,
            }
        )
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
                    use_split_k_matmul=use_split_k_matmul,
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
                use_tiled_prefill_attention=use_tiled_prefill_attention,
                use_native_online_decode=use_native_online_decode,
                use_fused_qk_norm_rope4=use_fused_qk_norm_rope4,
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

    def create_kv_cache(self, capacity: int | None = None) -> list[TinyKvCache]:
        from .kv_cache import TinyKvFullCache

        return [
            TinyKvFullCache(capacity=capacity) for _ in range(self.num_hidden_layers)
        ]

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
        submit_decode_prefix = (
            self.use_decode_submit18
            and type(self) is Qwen3ModelWeek2
            and self.num_hidden_layers == 36
            and inputs.shape == (1, 1)
            and self.precision == mx.bfloat16
            and len(cache) == 36
            and all(type(c) is TinyKvFullCache and c.uses_capacity for c in cache)
            and all(
                layer.self_attn.use_fused_qk_norm_rope4
                and layer.self_attn.use_native_online_decode
                for layer in self.layers_inner
            )
        )
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, rope_offsets, cache[layer], mask=mask)
            if submit_decode_prefix and layer == 17 and h.dtype == mx.bfloat16:
                tiny_llm_ext_ref_private._submit_decode_prefix_if_ordinary(h)
        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            h = h[:, -logits_to_keep:, :]
        h = self.norm(h)
        if self.w_lm_head is not None:
            return _linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)
