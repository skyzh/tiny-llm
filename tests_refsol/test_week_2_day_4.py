"""Week 2 Day 4 decode-attention tests."""

import mlx.core as mx

from tiny_llm_ref.attention import scaled_dot_product_attention_grouped
from .tiny_llm_base import (
    Qwen3ModelWeek2,
    RMSNorm,
    RoPE,
    decode_attention_custom,
    scaled_dot_product_attention,
)

from .utils import assert_allclose, tiny_qwen3_mlx_model


def test_model_integrates_decode_attention_before_fast_kernels():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="decode-attention")
    layer = model.layers_inner[0]

    assert layer.self_attn.use_decode_attention
    assert isinstance(layer.input_layernorm, RMSNorm)
    assert isinstance(layer.self_attn.rope, RoPE)
    assert not layer.mlp.use_fast_swiglu


def test_fast_attention_matches_grouped_attention():
    query = mx.random.normal((2, 4, 3, 16)).astype(mx.float32)
    key = mx.random.normal((2, 2, 5, 16)).astype(mx.float32)
    value = mx.random.normal((2, 2, 5, 16)).astype(mx.float32)
    mask = mx.broadcast_to(
        mx.array([0, 0, 0, 0, -mx.inf], dtype=mx.float32), (2, 1, 3, 5)
    )
    scale = 16**-0.5
    result = scaled_dot_product_attention(query, key, value, scale, mask)
    expected = scaled_dot_product_attention_grouped(query, key, value, scale, mask)
    assert result.shape == query.shape
    assert_allclose(result, expected, mx.float32, atol=1e-5, rtol=1e-5)


def test_custom_metal_attention_matches_grouped_attention():
    query = mx.random.normal((2, 4, 3, 16)).astype(mx.bfloat16)
    key = mx.random.normal((2, 2, 5, 16)).astype(mx.bfloat16)
    value = mx.random.normal((2, 2, 5, 16)).astype(mx.bfloat16)
    scale = 16**-0.5
    result = decode_attention_custom(query, key, value, scale, "causal")
    expected = scaled_dot_product_attention_grouped(query, key, value, scale, "causal")
    assert result.shape == query.shape
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)
