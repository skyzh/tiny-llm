from typing import Any

import mlx.core as mx

from .attention import flash_attention, paged_attention
from .embedding import QuantizedEmbedding
from .kv_cache import TinyKvCache
from .moe import Moe
from .paged_kv_cache import TinyKvPagedCache, TinyKvPagedPool
from .quantize import QuantizedWeights, quantized_linear
from .week2_kernels import (
    FastRMSNorm,
    FastRoPE,
    decode_attention_custom,
    scaled_dot_product_attention,
    swiglu,
)


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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.rope = FastRoPE(self.head_dim, max_seq_len, theta)
        self.q_norm = FastRMSNorm(self.head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = FastRMSNorm(self.head_dim, k_norm, eps=rms_norm_eps)
        self.use_flash_attention = use_flash_attention
        self.use_paged_attention = use_paged_attention

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        projection_q = quantized_linear(x, self.wq).reshape(
            B, L, self.num_heads, self.head_dim
        )
        projection_k = quantized_linear(x, self.wk).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        projection_q = self.q_norm(projection_q)
        projection_k = self.k_norm(projection_k)
        projection_v = quantized_linear(x, self.wv).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        projection_q = self.rope(projection_q, offset=offsets)
        projection_k = self.rope(projection_k, offset=offsets)
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)

        if self.use_flash_attention and L > 8:
            key, value, _, mask = cache.update_and_fetch(
                projection_k,
                projection_v,
                mask_length=L,
                mask=mask,
            )
            x = flash_attention(
                projection_q,
                key,
                value,
                scale=self.scale,
                mask=mask,
            )
        elif self.use_paged_attention:
            metadata = cache.update_and_fetch_paged(
                projection_k,
                projection_v,
                mask_length=L,
                mask=mask,
            )
            x = paged_attention(
                projection_q,
                metadata.key_pages,
                metadata.value_pages,
                metadata.block_table,
                metadata.context_lens,
                metadata.page_size,
                scale=self.scale,
                mask=metadata.mask,
            )
        else:
            key, value, _, mask = cache.update_and_fetch(
                projection_k,
                projection_v,
                mask_length=L,
                mask=mask,
            )
            if L <= 8 and key.shape[-2] <= 256:
                x = decode_attention_custom(
                    projection_q,
                    key,
                    value,
                    scale=self.scale,
                    mask=mask,
                )
            else:
                x = scaled_dot_product_attention(
                    projection_q,
                    key,
                    value,
                    scale=self.scale,
                    mask=mask,
                )
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return quantized_linear(x, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        return quantized_linear(
            swiglu(
                quantized_linear(x, self.w_gate),
                quantized_linear(x, self.w_up),
            ),
            self.w_down,
        )


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
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.mlp = mlp
        self.input_layernorm = FastRMSNorm(
            hidden_size, w_input_layernorm, eps=rms_norm_eps
        )
        self.post_attention_layernorm = FastRMSNorm(
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
            use_flash_attention=use_flash_attention,
            use_paged_attention=use_paged_attention,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), offset, cache, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


def is_qwen3_moe_sparse_layer(args: Any, layer_idx: int) -> bool:
    return (
        getattr(args, "num_experts", 0) > 0
        and layer_idx not in getattr(args, "mlp_only_layers", [])
        and (layer_idx + 1) % getattr(args, "decoder_sparse_step", 1) == 0
    )


class Qwen3ModelWeek3:
    def __init__(
        self,
        mlx_model: Any,
        page_size: int = 128,
        enable_flash_attn: bool = False,
        enable_performance_lab: bool = False,
        enable_paged_attention: bool = True,
    ):
        if enable_flash_attn and mlx_model.args.head_dim != 128:
            raise ValueError("Week 3 FlashAttention requires head_dim=128")
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        self.page_size = page_size
        # Each layer owns physical storage shared by all request caches for that
        # layer. Page ids are therefore layer-local, as they are in the kernel.
        self.page_pools = [
            TinyKvPagedPool(page_size=self.page_size)
            for _ in range(self.num_hidden_layers)
        ]
        precision = mx.bfloat16
        self.precision = precision

        def week3_weights(layer: Any) -> QuantizedWeights:
            return QuantizedWeights.from_mlx_layer(
                layer, use_simdgroup_matmul=enable_performance_lab
            )

        self.embedding = QuantizedEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=week3_weights(mlx_model.model.embed_tokens),
            use_custom_kernel=enable_performance_lab,
        )
        self.layers_inner = []

        for i in range(mlx_model.args.num_hidden_layers):
            wq = week3_weights(mlx_model.model.layers[i].self_attn.q_proj)
            wk = week3_weights(mlx_model.model.layers[i].self_attn.k_proj)
            wv = week3_weights(mlx_model.model.layers[i].self_attn.v_proj)
            wo = week3_weights(mlx_model.model.layers[i].self_attn.o_proj)
            if is_qwen3_moe_sparse_layer(mlx_model.args, i):
                mlp = Moe(
                    w_router=week3_weights(mlx_model.model.layers[i].mlp.gate),
                    w_gate=week3_weights(
                        mlx_model.model.layers[i].mlp.switch_mlp.gate_proj
                    ),
                    w_up=week3_weights(
                        mlx_model.model.layers[i].mlp.switch_mlp.up_proj
                    ),
                    w_down=week3_weights(
                        mlx_model.model.layers[i].mlp.switch_mlp.down_proj
                    ),
                    num_experts_per_tok=mlx_model.args.num_experts_per_tok,
                    norm_topk_prob=mlx_model.args.norm_topk_prob,
                )
            else:
                mlp = Qwen3MLP(
                    mlx_model.args.hidden_size,
                    mlx_model.args.intermediate_size,
                    week3_weights(mlx_model.model.layers[i].mlp.gate_proj),
                    week3_weights(mlx_model.model.layers[i].mlp.up_proj),
                    week3_weights(mlx_model.model.layers[i].mlp.down_proj),
                )

            layer = Qwen3TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                head_dim=mlx_model.args.head_dim,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                q_norm=mlx_model.model.layers[i].self_attn.q_norm.weight,
                k_norm=mlx_model.model.layers[i].self_attn.k_norm.weight,
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight,
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight,
                mlp=mlp,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
                use_flash_attention=enable_flash_attn,
                use_paged_attention=enable_paged_attention,
            )
            self.layers_inner.append(layer)
        self.norm = FastRMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight,
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = week3_weights(mlx_model.lm_head)
        else:
            self.w_lm_head = None
        self.mlx_model = mlx_model

    def create_kv_cache(self) -> list[TinyKvCache]:
        # One request gets one cache handle per layer. Requests share storage
        # within a layer, while every handle keeps independent logical metadata.
        return [TinyKvPagedCache(pool=pool) for pool in self.page_pools]

    def __call__(
        self,
        inputs: mx.array,
        offset: int | list[int] | mx.array,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        h = self.embedding(inputs)
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, offset, cache[layer], mask="causal")
        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            h = h[:, -logits_to_keep:, :]
        h = self.norm(h)
        if self.w_lm_head is not None:
            return quantized_linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)
