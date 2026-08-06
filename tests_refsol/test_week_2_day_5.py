"""Week 2 Day 5 decode-attention tests."""

from math import prod

import mlx.core as mx
import pytest

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

    cases = (
        (0, 1, "custom", 1),
        (29, 2, "custom", 31),
        (126, 2, "custom", 128),
        (254, 2, "custom", 256),
        (256, 1, "readable", 257),
        (253, 3, "readable", 256),
        (248, 8, "readable", 256),
    )
    for prefix_length, query_length, expected_path, expected_context in cases:
        model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="decode-attention")
        attention = model.layers_inner[0].self_attn
        cache = model.create_kv_cache()[0]
        hidden = model.hidden_size
        if prefix_length:
            mx.eval(
                attention(
                    mx.zeros((1, prefix_length, hidden), dtype=model.precision),
                    0,
                    cache,
                )
            )
        calls.clear()
        mx.eval(
            attention(
                mx.zeros((1, query_length, hidden), dtype=model.precision),
                prefix_length,
                cache,
            )
        )
        assert calls == [(expected_path, expected_context)]


def test_model_keeps_explicit_masks_on_readable_path(monkeypatch):
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="decode-attention")
    attention = model.layers_inner[0].self_attn
    cache = model.create_kv_cache()[0]
    module = __import__(Qwen3ModelWeek2.__module__, fromlist=["unused"])
    readable_attention = module.scaled_dot_product_attention_grouped
    calls = []

    def reject_custom(*args, **kwargs):
        pytest.fail("explicit masks must not use the bounded decode kernel")

    def record_readable(query, key, value, scale, mask):
        calls.append(key.shape[-2])
        return readable_attention(query, key, value, scale, mask)

    monkeypatch.setattr(module, "decode_attention_custom", reject_custom)
    monkeypatch.setattr(module, "scaled_dot_product_attention_grouped", record_readable)

    hidden = mx.zeros((1, 1, model.hidden_size), dtype=model.precision)
    mask = mx.zeros((1, 1, 1, 1), dtype=mx.float32)
    mx.eval(attention(hidden, 0, cache, mask))

    assert calls == [1]


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


def test_custom_metal_attention_matches_qwen_boundary_sweep():
    head_dim = 128
    query_heads = 4
    shapes = (
        *((1, context) for context in (1, 31, 32, 127, 128, 129, 255, 256)),
        *((8, context) for context in (8, 31, 32, 127, 128, 129, 255, 256)),
    )

    def fixture(shape, phase):
        values = mx.sin(
            mx.arange(prod(shape), dtype=mx.float32) * 0.017 + phase
        ).reshape(shape)
        return values.astype(mx.bfloat16)

    for query_length, context_length in shapes:
        for gqa_ratio in (1, 4):
            kv_heads = query_heads // gqa_ratio
            query = fixture((1, query_heads, query_length, head_dim), 0.1)
            key = fixture((1, kv_heads, context_length, head_dim), 0.7)
            value = fixture(key.shape, 1.3)
            explicit_mask = mx.where(
                mx.arange(context_length) % 5 == 0,
                mx.array(-2.0, dtype=mx.float32),
                mx.array(0.0, dtype=mx.float32),
            ).reshape(1, 1, 1, context_length)

            for mask in ("causal", explicit_mask):
                result = decode_attention_custom(
                    query, key, value, head_dim**-0.5, mask
                )
                expected = scaled_dot_product_attention_grouped(
                    query, key, value, head_dim**-0.5, mask
                )
                assert result.shape == query.shape
                assert_allclose(
                    result,
                    expected,
                    mx.bfloat16,
                    atol=3e-2,
                    rtol=3e-2,
                    message=(
                        f"L={query_length}, S={context_length}, "
                        f"GQA={gqa_ratio}, mask={type(mask).__name__}"
                    ),
                )


def test_custom_metal_attention_rejects_unknown_string_mask():
    query = mx.zeros((1, 4, 1, 128), dtype=mx.bfloat16)
    key = mx.zeros((1, 1, 1, 128), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="unsupported attention mask"):
        decode_attention_custom(query, key, key, 128**-0.5, "sliding")
