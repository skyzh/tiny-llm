import mlx.core as mx

from .basics import silu
from .attention import paged_attention
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights, quantized_linear
from .kv_cache import TinyKvCache
from .paged_kv_cache import TinyKvPagedCache, TinyKvPagedPool
from .qwen3_week2 import Qwen3MLP


def assert_dtype(weights: mx.array, dtype: mx.Dtype):
    if weights.dtype != dtype:
        raise ValueError(f"{weights.dtype} != {dtype}")
    else:
        return weights


def assert_quantized_weights_dtype(weights: QuantizedWeights, dtype: mx.Dtype):
    if weights.scales.dtype != dtype:
        raise ValueError(f"{weights.scales.dtype} != {dtype}")
    if weights.biases.dtype != dtype:
        raise ValueError(f"{weights.biases.dtype} != {dtype}")
    else:
        return weights


class Qwen3PagedMultiHeadAttention:
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
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.head_dim = head_dim
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.rope = RoPE(self.head_dim, max_seq_len, theta)
        self.q_norm = RMSNorm(self.head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, k_norm, eps=rms_norm_eps)

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
        if isinstance(offsets, int):
            offset_slice = [slice(int(offsets), int(offsets + L))]
        else:
            offset_slice = [slice(int(i), int(i + L)) for i in offsets]
        projection_q = self.rope(projection_q, offset=offset_slice)
        projection_k = self.rope(projection_k, offset=offset_slice)
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)

        metadata = cache.update_and_fetch_paged(
            projection_k.astype(mx.float32),
            projection_v.astype(mx.float32),
            mask_length=L,
            mask=mask,
        )
        x = paged_attention(
            projection_q.astype(mx.float32),
            metadata.key_pages,
            metadata.value_pages,
            metadata.block_table,
            metadata.context_lens,
            metadata.page_size,
            scale=self.scale,
            mask=metadata.mask,
        ).astype(x.dtype)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return quantized_linear(x, self.wo)


class Qwen3TransformerBlockWeek3:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        self.self_attn = Qwen3PagedMultiHeadAttention(
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


class Qwen3ModelWeek3:
    def __init__(
        self,
        mlx_model: Any,
        page_size: int = 128,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        self.page_size = page_size
        self.page_pool = TinyKvPagedPool(page_size=self.page_size)
        precision = mx.bfloat16
        self.precision = precision

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=assert_dtype(
                dequantize_linear(mlx_model.model.embed_tokens), dtype=precision
            ),
        )
        self.layers_inner = []

        for i in range(mlx_model.args.num_hidden_layers):
            wq = assert_quantized_weights_dtype(
                QuantizedWeights.from_mlx_layer(
                    mlx_model.model.layers[i].self_attn.q_proj
                ),
                dtype=precision,
            )
            wk = assert_quantized_weights_dtype(
                QuantizedWeights.from_mlx_layer(
                    mlx_model.model.layers[i].self_attn.k_proj
                ),
                dtype=precision,
            )
            wv = assert_quantized_weights_dtype(
                QuantizedWeights.from_mlx_layer(
                    mlx_model.model.layers[i].self_attn.v_proj
                ),
                dtype=precision,
            )
            wo = assert_quantized_weights_dtype(
                QuantizedWeights.from_mlx_layer(
                    mlx_model.model.layers[i].self_attn.o_proj
                ),
                dtype=precision,
            )
            w_gate = assert_quantized_weights_dtype(
                QuantizedWeights.from_mlx_layer(
                    mlx_model.model.layers[i].mlp.gate_proj
                ),
                dtype=precision,
            )
            w_up = assert_quantized_weights_dtype(
                QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].mlp.up_proj),
                dtype=precision,
            )
            w_down = assert_quantized_weights_dtype(
                QuantizedWeights.from_mlx_layer(
                    mlx_model.model.layers[i].mlp.down_proj
                ),
                dtype=precision,
            )

            layer = Qwen3TransformerBlockWeek3(
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
                q_norm=assert_dtype(
                    mlx_model.model.layers[i].self_attn.q_norm.weight, dtype=precision
                ),
                k_norm=assert_dtype(
                    mlx_model.model.layers[i].self_attn.k_norm.weight, dtype=precision
                ),
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=assert_dtype(
                    mlx_model.model.layers[i].input_layernorm.weight, dtype=precision
                ),
                w_post_attention_layernorm=assert_dtype(
                    mlx_model.model.layers[i].post_attention_layernorm.weight,
                    dtype=precision,
                ),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=assert_dtype(mlx_model.model.norm.weight, dtype=precision),
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = assert_quantized_weights_dtype(
                QuantizedWeights.from_mlx_layer(mlx_model.lm_head), dtype=precision
            )
        else:
            self.w_lm_head = None
        self.mlx_model = mlx_model

    def create_kv_cache(self) -> list[TinyKvCache]:
        return [
            TinyKvPagedCache(pool=self.page_pool) for _ in range(self.num_hidden_layers)
        ]

    def __call__(
        self,
        inputs: mx.array,
        offset: int | list[int] | mx.array,
        cache: list[TinyKvCache],
    ) -> mx.array:
        h = self.embedding(inputs)
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, offset, cache[layer], mask="causal")
        h = self.norm(h)
        if self.w_lm_head is not None:
            return quantized_linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)
