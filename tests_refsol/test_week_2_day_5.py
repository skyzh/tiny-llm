"""Week 2 Day 5 decode-attention tests."""

import mlx.core as mx

from tiny_llm_ref.attention import scaled_dot_product_attention_grouped
from .tiny_llm_base import (
    FastRMSNorm,
    FastRoPE,
    Qwen3ModelWeek2,
    decode_attention_custom,
    scaled_dot_product_attention,
)

from .utils import assert_allclose, tiny_qwen3_mlx_model


def test_model_integrates_decode_attention_after_fast_kernels():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="decode-attention")
    layer = model.layers_inner[0]

    assert layer.self_attn.use_decode_attention
    assert isinstance(layer.input_layernorm, FastRMSNorm)
    assert isinstance(layer.self_attn.rope, FastRoPE)
    assert layer.mlp.use_fast_swiglu


def test_model_uses_decode_attention_only_through_measured_context(monkeypatch):
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="decode-attention")
    attention = model.layers_inner[0].self_attn
    cache = model.create_kv_cache()[0]
    module = __import__(Qwen3ModelWeek2.__module__, fromlist=["unused"])
    readable_attention = module.scaled_dot_product_attention_grouped
    calls = []

    def record_custom(query, key, value, scale, mask):
        calls.append(("custom", key.shape[-2]))
        return mx.zeros_like(query)

    def record_readable(query, key, value, scale, mask):
        calls.append(("readable", key.shape[-2]))
        return readable_attention(query, key, value, scale, mask)

    monkeypatch.setattr(module, "decode_attention_custom", record_custom)
    monkeypatch.setattr(module, "scaled_dot_product_attention_grouped", record_readable)

    hidden = model.hidden_size
    mx.eval(attention(mx.zeros((1, 127, hidden), dtype=model.precision), 0, cache))
    calls.clear()
    mx.eval(attention(mx.zeros((1, 1, hidden), dtype=model.precision), 127, cache))
    mx.eval(attention(mx.zeros((1, 1, hidden), dtype=model.precision), 128, cache))

    assert calls == [("custom", 128), ("readable", 129)]


def test_fast_attention_matches_grouped_attention():
    query = mx.random.normal((2, 4, 3, 16)).astype(mx.bfloat16)
    key = mx.random.normal((2, 2, 5, 16)).astype(mx.bfloat16)
    value = mx.random.normal((2, 2, 5, 16)).astype(mx.bfloat16)
    mask = mx.broadcast_to(
        mx.array([0, 0, 0, 0, -mx.inf], dtype=mx.bfloat16), (2, 1, 3, 5)
    )
    scale = 16**-0.5
    result = scaled_dot_product_attention(query, key, value, scale, mask)
    expected = scaled_dot_product_attention_grouped(query, key, value, scale, mask)
    assert result.shape == query.shape
    assert result.dtype == mx.bfloat16
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


def test_custom_metal_attention_matches_grouped_attention():
    query = mx.random.normal((2, 4, 3, 16)).astype(mx.bfloat16)
    key = mx.random.normal((2, 2, 5, 16)).astype(mx.bfloat16)
    value = mx.random.normal((2, 2, 5, 16)).astype(mx.bfloat16)
    scale = 16**-0.5
    result = decode_attention_custom(query, key, value, scale, "causal")
    expected = scaled_dot_product_attention_grouped(query, key, value, scale, "causal")
    assert result.shape == query.shape
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)
