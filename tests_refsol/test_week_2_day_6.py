"""Week 2 Day 6 tiled dense-prefill tests."""

from math import prod

import mlx.core as mx

from .tiny_llm_base import (
    Qwen3ModelWeek2,
    dense_prefill_attention_mma,
    scaled_dot_product_attention,
)
from .utils import assert_allclose, tiny_qwen3_mlx_model


def _fixture(shape: tuple[int, ...], phase: float) -> mx.array:
    values = mx.sin(mx.arange(prod(shape), dtype=mx.float32) * 0.017 + phase)
    return values.reshape(shape).astype(mx.bfloat16)


def test_task_1_tiled_prefill_matches_readable_causal_gqa():
    query = _fixture((1, 4, 33, 128), 0.1)
    key = _fixture((1, 2, 47, 128), 0.7)
    value = _fixture(key.shape, 1.3)

    actual = dense_prefill_attention_mma(query, key, value, 128**-0.5, "causal")
    expected = scaled_dot_product_attention(
        query.astype(mx.float32),
        key.astype(mx.float32),
        value.astype(mx.float32),
        128**-0.5,
        "causal",
    ).astype(mx.bfloat16)

    assert actual.shape == query.shape
    assert actual.dtype == mx.bfloat16
    assert_allclose(actual, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


def test_task_2_fully_masked_rows_are_finite_zero():
    query = _fixture((1, 4, 9, 128), 0.1)
    key = _fixture((1, 1, 17, 128), 0.7)
    value = _fixture(key.shape, 1.3)
    mask = mx.full((1, 1, 9, 17), -mx.inf, dtype=mx.float32)

    actual = dense_prefill_attention_mma(query, key, value, 128**-0.5, mask)
    mx.eval(actual)
    assert bool(mx.all(mx.isfinite(actual)).item())
    assert float(mx.max(mx.abs(actual)).item()) == 0.0


def test_task_3_tiled_prefill_checkpoint_runs_the_week2_engine():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="tiled-prefill")
    layer = model.layers_inner[0]

    assert model.use_bounded_kv_capacity
    assert layer.self_attn.use_tiled_prefill_attention
    assert not hasattr(layer.self_attn, "use_decode_attention")

    output = model(
        mx.array([[1, 2, 3]], dtype=mx.int32),
        0,
        model.create_kv_cache(capacity=3),
    )
    assert output.dtype == mx.bfloat16
    assert layer.self_attn.attention_dispatch_counts["tiled_prefill_fallback"] == 1
    assert layer.self_attn.attention_dispatch_counts["readable"] == 1
