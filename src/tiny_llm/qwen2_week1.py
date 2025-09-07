import mlx.core as mx
from mlx.nn import quantize
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear
from tiny_llm import embedding


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
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        
        self.RoPE = RoPE(self.head_dim, max_seq_len, theta)

        assert (self.num_heads > 0 and self.hidden_size % self.num_heads == 0)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, E = x.shape
        query = linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, -1)
        key = linear(x, self.wk, self.bk).reshape(B, L, self.num_kv_heads, -1)
        value = linear(x, self.wv, self.bv).reshape(B, L, self.num_kv_heads, -1)
        
        query = self.RoPE(query, offset=slice(0, L))
        key = self.RoPE(key, offset=slice(0, L))

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        
        attn_output = scaled_dot_product_attention_grouped(
            query.astype(mx.float32), 
            key.astype(mx.float32), 
            value.astype(mx.float32), 
            scale=self.scale, 
            mask=mask
        ).astype(x.dtype)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, E)
        
        output = linear(attn_output, self.wo)
        return output

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

        assert (self.w_gate.shape == (hidden_dim, dim))
        assert (self.w_up.shape == (hidden_dim, dim))
        assert (self.w_down.shape == (dim, hidden_dim))

    def __call__(self, x: mx.array) -> mx.array:
        assert x.shape[-1] == self.dim
        x_up = linear(x, self.w_up)
        x_glu = silu(linear(x, self.w_gate))
        x = x_up * x_glu
        
        return linear(x, self.w_down)


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
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.self_attn = Qwen2MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            wq, wk, wv, wo, bq, bk, bv,
            max_seq_len,
            theta
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, rms_norm_eps
        )
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask)
        
        x = h + x
        
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)

        x = h + x
        return x


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        hidden_size = mlx_model.args.hidden_size
        vocab_size = mlx_model.args.vocab_size

        self.precision = mx.float32
        
        embedding_weight = dequantize_linear(mlx_model.model.embed_tokens).astype(self.precision)
        self.embedding = Embedding(vocab_size, hidden_size, embedding_weight)
        
        num_hidden_layers = mlx_model.args.num_hidden_layers

        self.hidden_laysers = []
        for i in range(num_hidden_layers):
            tb = mlx_model.layers[i]
            wq = dequantize_linear(tb.self_attn.q_proj).astype(self.precision)
            wk = dequantize_linear(tb.self_attn.k_proj).astype(self.precision)
            wv = dequantize_linear(tb.self_attn.v_proj).astype(self.precision)
            wo = dequantize_linear(tb.self_attn.o_proj).astype(self.precision)

            bq = tb.self_attn.q_proj.bias.astype(self.precision)
            bk = tb.self_attn.k_proj.bias.astype(self.precision)
            bv = tb.self_attn.v_proj.bias.astype(self.precision)
            
            w_gate = dequantize_linear(tb.mlp.gate_proj).astype(self.precision)
            w_up = dequantize_linear(tb.mlp.up_proj).astype(self.precision)
            w_down = dequantize_linear(tb.mlp.down_proj).astype(self.precision)

            w_input_layernorm = tb.input_layernorm.weight.astype(self.precision)
            w_post_atten_layernorm = tb.post_attention_layernorm.weight.astype(self.precision)
            
            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                bq=bq,
                bk=bk,
                bv=bv,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=w_input_layernorm,
                w_post_attention_layernorm=w_post_atten_layernorm,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta
            )
            self.hidden_laysers.append(layer)
        
        self.norm = RMSNorm(
            dim=mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(self.precision),
            eps=mlx_model.args.rms_norm_eps
        )

        if not mlx_model.args.tie_word_embeddings:
            self.lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.lm_head = None

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        h = self.embedding(inputs)
        for layer in self.hidden_laysers:
            h = layer(h, mask="causal")
        
        h = self.norm(h)
        
        if self.lm_head is not None:
            return linear(h, self.lm_head)
        else:
            return self.embedding.as_linear(h)