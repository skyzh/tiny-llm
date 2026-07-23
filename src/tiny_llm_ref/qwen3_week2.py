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
    decode_attention_custom,
    swiglu,
)

WEEK2_CHECKPOINTS = (
    "kv-cache",
    "quantized-matvec",
    "decode-attention",
    "rmsnorm",
    "rope",
    "swiglu",
    "simd-matmul",
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
        use_decode_attention: bool = True,
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
        self.use_decode_attention = use_decode_attention
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
        projection_q = self.rope(projection_q, offset=rope_offsets)
        projection_k = self.rope(projection_k, offset=rope_offsets)
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)
        projection_k, projection_v, _, mask = cache.update_and_fetch(
            projection_k, projection_v, mask_length=L, mask=mask
        )
        if self.use_decode_attention and L <= 8 and projection_k.shape[-2] <= 256:
            x = decode_attention_custom(
                projection_q,
                projection_k,
                projection_v,
                scale=self.scale,
                mask=mask,
            )
        else:
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
        use_decode_attention: bool = True,
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
            use_decode_attention=use_decode_attention,
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
    def __init__(self, mlx_model: Any, checkpoint: str = "simd-matmul"):
        if checkpoint not in WEEK2_CHECKPOINTS:
            raise ValueError(
                f"unknown Week 2 checkpoint {checkpoint!r}; "
                f"choose one of {WEEK2_CHECKPOINTS}"
            )
        checkpoint_index = WEEK2_CHECKPOINTS.index(checkpoint)
        self.checkpoint = checkpoint
        use_quantized_weights = checkpoint_index >= WEEK2_CHECKPOINTS.index(
            "quantized-matvec"
        )
        use_fast_rms_norm = checkpoint_index >= WEEK2_CHECKPOINTS.index("rmsnorm")
        use_fast_rope = checkpoint_index >= WEEK2_CHECKPOINTS.index("rope")
        use_fast_swiglu = checkpoint_index >= WEEK2_CHECKPOINTS.index("swiglu")
        use_decode_attention = checkpoint_index >= WEEK2_CHECKPOINTS.index(
            "decode-attention"
        )
        use_simdgroup_matmul = checkpoint_index >= WEEK2_CHECKPOINTS.index(
            "simd-matmul"
        )
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.use_fast_rope = use_fast_rope
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        precision = mx.bfloat16
        self.precision = precision

        def model_weight(layer: Any) -> mx.array | QuantizedWeights:
            if use_quantized_weights:
                return QuantizedWeights.from_mlx_layer(
                    layer, use_simdgroup_matmul=use_simdgroup_matmul
                )
            return dequantize_linear(layer).astype(mx.bfloat16)

        embedding_weight = model_weight(mlx_model.model.embed_tokens)
        if isinstance(embedding_weight, QuantizedWeights):
            self.embedding = QuantizedEmbedding(
                vocab_size=self.vocab_size,
                embedding_dim=self.hidden_size,
                weight=embedding_weight,
                use_custom_kernel=use_simdgroup_matmul,
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
                use_decode_attention=use_decode_attention,
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
