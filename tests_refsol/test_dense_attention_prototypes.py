"""Focused model-free controls for tiled dense-prefill attention."""

from math import prod

import mlx.core as mx
import pytest

from tiny_llm_ref.week2_kernels import (
    dense_prefill_attention_mma,
    scaled_dot_product_attention,
)
from .utils import assert_allclose


def _fixture(shape: tuple[int, ...], phase: float, dtype=mx.bfloat16) -> mx.array:
    values = mx.sin(mx.arange(prod(shape), dtype=mx.float32) * 0.017 + phase)
    return values.reshape(shape).astype(dtype)


def _explicit_mask(length: int, context: int) -> mx.array:
    values = mx.where(
        mx.arange(context) % 5 == 0,
        mx.array(-1.25, dtype=mx.float32),
        mx.array(0.0, dtype=mx.float32),
    )
    return mx.broadcast_to(values.reshape(1, 1, 1, context), (1, 1, length, context))


@pytest.mark.parametrize(
    ("length", "context", "gqa_ratio", "mask_kind"),
    (
        (9, 9, 1, "none"),
        (15, 16, 2, "causal"),
        (16, 31, 4, "explicit"),
        (31, 32, 1, "causal"),
        (32, 33, 2, "explicit"),
        (33, 47, 4, "none"),
    ),
)
def test_tiled_prefill_matches_readable_boundary_sweep(
    length: int,
    context: int,
    gqa_ratio: int,
    mask_kind: str,
):
    query_heads = 4
    kv_heads = query_heads // gqa_ratio
    query = _fixture((1, query_heads, length, 128), 0.1)
    key = _fixture((1, kv_heads, context, 128), 0.7)
    value = _fixture(key.shape, 1.3)
    mask = {
        "none": None,
        "causal": "causal",
        "explicit": _explicit_mask(length, context),
    }[mask_kind]

    actual = dense_prefill_attention_mma(query, key, value, 128**-0.5, mask)
    expected = scaled_dot_product_attention(
        query.astype(mx.float32),
        key.astype(mx.float32),
        value.astype(mx.float32),
        128**-0.5,
        mask,
    ).astype(mx.bfloat16)

    assert actual.shape == query.shape
    assert actual.dtype == mx.bfloat16
    assert_allclose(
        actual,
        expected,
        mx.bfloat16,
        atol=2e-2,
        rtol=2e-2,
        message=f"L={length}, S={context}, GQA={gqa_ratio}, mask={mask_kind}",
    )


def test_tiled_prefill_fully_masked_rows_are_finite_zero_and_contract_is_narrow():
    query = _fixture((1, 4, 9, 128), 0.1)
    key = _fixture((1, 2, 17, 128), 0.7)
    value = _fixture(key.shape, 1.3)
    mask = mx.full((1, 1, 9, 17), -mx.inf, dtype=mx.float32)

    actual = dense_prefill_attention_mma(query, key, value, 128**-0.5, mask)
    mx.eval(actual)
    assert bool(mx.all(mx.isfinite(actual)).item())
    assert float(mx.max(mx.abs(actual)).item()) == 0.0

    with pytest.raises(ValueError, match="L >= 9"):
        dense_prefill_attention_mma(query[:, :, :8], key, value, 128**-0.5)
    with pytest.raises(ValueError, match="BF16"):
        dense_prefill_attention_mma(
            query.astype(mx.float32),
            key.astype(mx.float32),
            value.astype(mx.float32),
            128**-0.5,
        )


@pytest.mark.parametrize("context", (2048, 8192))
def test_tiled_prefill_long_context_shape_smoke(context: int):
    query = _fixture((1, 1, context, 128), 0.1)
    key = _fixture((1, 1, context, 128), 0.7)
    value = _fixture(key.shape, 1.3)
    actual = dense_prefill_attention_mma(query, key, value, 128**-0.5)
    mx.eval(actual)
    assert actual.shape == query.shape
    assert bool(mx.all(mx.isfinite(actual)).item())
