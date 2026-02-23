import mlx.core as mx

from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, quantized_linear, QuantizedWeights
from .kv_cache import TinyKvCache


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        offsets: list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        query = quantized_linear(x, self.wq, bias=self.bq).reshape(B, L, self.num_heads, self.head_dim)
        key = quantized_linear(x, self.wk, bias=self.bk).reshape(B, L, self.num_kv_heads, self.head_dim)
        value = quantized_linear(x, self.wv, bias=self.bv).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = list(map(lambda x: slice(x, x+L), offsets))
        query = self.rope(query, offset).transpose(0, 2, 1, 3)
        key = self.rope(key, offset).transpose(0, 2, 1, 3)

        key, value, _, _ = cache.update_and_fetch(key, value)

        # TODO: why use float32 to compute 
        output = scaled_dot_product_attention_grouped(
            query.astype(mx.float32),
            key.astype(mx.float32),
            value.astype(mx.float32),
            scale=None,
            mask=mask
        ).astype(x.dtype).transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        output = quantized_linear(output, self.wo)
        return output

class Qwen2MLP:
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
        gate = silu(quantized_linear(x, self.w_gate))
        value = quantized_linear(x, self.w_up)

        return quantized_linear(gate * value, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, rms_norm_eps)
        self.attention = Qwen2MultiHeadAttention(hidden_size, num_attention_heads, num_kv_heads, wq, wk, wv, wo, bq, bk, bv, max_seq_len, theta)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        
    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        offsets = [offset] * x.shape[0]  
        r = self.attention(self.input_layernorm(x), offsets, cache, mask)
        h = r + x
        r = self.mlp(self.post_attention_layernorm(h))
        return r + h


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        model = mlx_model.model
        args = mlx_model.args
        self.precision = mx.float16

        self.embedding = Embedding(args.vocab_size, args.hidden_size, weight=dequantize_linear(model.embed_tokens)) 
        self.layers = []
        for layer in model.layers:
            wq = QuantizedWeights.from_mlx_layer(layer.self_attn.q_proj)
            wk = QuantizedWeights.from_mlx_layer(layer.self_attn.k_proj)
            wv = QuantizedWeights.from_mlx_layer(layer.self_attn.v_proj)
            wo = QuantizedWeights.from_mlx_layer(layer.self_attn.o_proj)
            w_gate = QuantizedWeights.from_mlx_layer(layer.mlp.gate_proj)
            w_up = QuantizedWeights.from_mlx_layer(layer.mlp.up_proj)
            w_down = QuantizedWeights.from_mlx_layer(layer.mlp.down_proj)

            self.layers.append(Qwen2TransformerBlock(
                args.num_attention_heads, args.num_key_value_heads,
                args.hidden_size, args.intermediate_size, args.rms_norm_eps,
                wq, wk, wv, wo,
                layer.self_attn.q_proj.bias.astype(self.precision), layer.self_attn.k_proj.bias.astype(self.precision), layer.self_attn.v_proj.bias.astype(self.precision),
                w_gate, w_up, w_down,
                layer.input_layernorm.weight.astype(self.precision), layer.post_attention_layernorm.weight.astype(self.precision),
                args.max_position_embeddings, args.rope_theta
            ))
        
        if not args.tie_word_embeddings:
            self.w_lm_head = QuantizedWeights.from_mlx_layer(mlx_model.lm_head)
        else:
            self.w_lm_head = None
    
        self.rms_norm = RMSNorm(args.hidden_size, model.norm.weight.astype(self.precision))

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        x = self.embedding(inputs)
        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, offset, layer_cache, mask="causal")
        x = self.rms_norm(x)

        if self.w_lm_head is None:
            return self.embedding.as_linear(x)
        else:
            return quantized_linear(x, self.w_lm_head)
