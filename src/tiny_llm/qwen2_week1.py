import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


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
        self.d_k = hidden_size // num_heads
        self.rope = RoPE(self.d_k, max_seq_len, theta, False)   # use untroditional RoPE
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 1. liear
        N, L, _ = x.shape   # x: (N, L, d_model)
        query = (
            linear(x, self.wq, self.bq)  # query: (N, L, d_model)
            .reshape(N, L, self.num_heads, self.d_k)    # query: (N, L, h, d_k)
            # .transpose(0, 2, 1, 3)  # query: (N, h, L, d_k)
        )
        key = (
            linear(x, self.wk, self.bk)
            .reshape(N, L, self.num_kv_heads, self.d_k)
        )
        value = (
            linear(x, self.wv, self.bv)
            .reshape(N, L, self.num_kv_heads, self.d_k)
        )

        # 2. RoPE
        query = self.rope(query, offset=slice(0, L))
        key = self.rope(key, offset=slice(0, L))
        query = query.transpose(0, 2, 1, 3)  # query: (N, h, L, d_k)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        # 2-3. scaled dot-product attention & concat
        scale = 1.0 / mx.sqrt(self.d_k)   # scale = 1 / sqrt(d_k)
        scores = (
            scaled_dot_product_attention_grouped(query, key, value, scale, mask=mask)  # scores: (N, h, L, d_k)
            .transpose(0, 2, 1, 3)  # scores: (N, L, h, d_k)
            .reshape(N, L, self.hidden_size)  # scores: (N, L, d_model)
        )

        # 4. linear
        return linear(scores, self.wo)  # output: (N, L, d_model)


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

    def __call__(self, x: mx.array) -> mx.array:
        # mlp(x) = ((silu(x * w_gate.T) @ (x * w_up.T))) @ w_down.T
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


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
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, eps=rms_norm_eps)
        self.self_attn = Qwen2MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            wq, wk, wv, wo,
            bq, bk, bv,
            max_seq_len,
            theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 1. input layer norm
        x_norm = self.input_layernorm(x)

        # 2. self attention
        x = self.self_attn(x_norm, mask) + x

        # 3. post attention layer norm
        x_norm = self.post_attention_layernorm(x)

        # 4. mlp
        x = self.mlp(x_norm) + x

        return x


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        precision = mx.float16
        self.hidden_size = mlx_model.args.hidden_size

        # 1. embedding
        self.embeddings = Embedding(
            vocab_size=mlx_model.args.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(precision),
        )

        # 2. transformer blocks
        self.layers_inner = []
        for i in range(mlx_model.args.num_hidden_layers):
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj).astype(precision)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj).astype(precision)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj).astype(precision)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj).astype(precision)
            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj).astype(precision)
            w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj).astype(precision)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj).astype(precision)
            
            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=self.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(precision),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(precision),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(precision),
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight.astype(precision),
                w_post_attention_layernorm=mlx_model.model.layers[i].post_attention_layernorm.weight.astype(precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)
        
        # 3. RMSNorm final
        self.norm = RMSNorm(
            self.hidden_size,
            weight=mlx_model.model.norm.weight.astype(precision),
            eps=mlx_model.args.rms_norm_eps,
        )

        # 4. Embedding::as_linear OR Linear (lm_head)
        if not mlx_model.args.tie_word_embeddings:
            self.lm_head = dequantize_linear(mlx_model.lm_head).astype(precision)
        else:
            self.lm_head = None

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        # 1. embedding
        x = self.embeddings(inputs)  # (N, L, d_model)

        # 2. transformer blocks
        for layer in self.layers_inner:
            x = layer(x, mask="causal")  # (N, L, d_model)
        
        # 3. RMSNorm final
        x = self.norm(x)  # (N, L, d_model)

        # 4. Embedding::as_linear OR Linear (lm_head)
        if self.lm_head is not None:
            return linear(x, self.lm_head)  # untied weights
        else:
            return self.embeddings.as_linear(x) # tied weights