"""Reference-only regression coverage; these are not additional learner tasks."""

from math import prod

import mlx.core as mx
import pytest

from tiny_llm_ref.attention import scaled_dot_product_attention_grouped
from tiny_llm_ref.qwen3_week2 import Qwen3ModelWeek2
from tiny_llm_ref.quantize import quantized_matmul
from tiny_llm_ref.week2_kernels import (
    FastRMSNorm,
    FastRoPE,
    _qk_norm_rope4,
    decode_attention_online,
)

from .utils import assert_allclose, tiny_qwen3_mlx_model


def _fixture(shape, phase):
    return (
        mx.sin(mx.arange(prod(shape), dtype=mx.float32) * 0.017 + phase)
        .reshape(shape)
        .astype(mx.bfloat16)
    )


@pytest.mark.parametrize("context", (256, 257, 8192))
def test_online_decode_matches_readable_with_padded_cache(context):
    """Cover the serial/split boundary and grouped long-context route."""
    query = _fixture((1, 8, 1, 128), 0.1)
    key = _fixture((1, 2, context + 17, 128), 0.7)[:, :, :context, :]
    value = _fixture((1, 2, context + 17, 128), 1.3)[:, :, :context, :]
    mask = mx.where(mx.arange(context) % 7 == 0, -mx.inf, 0.0)
    actual = decode_attention_online(query, key, value, 128**-0.5, mask)
    expected = scaled_dot_product_attention_grouped(
        query.astype(mx.float32),
        key.astype(mx.float32),
        value.astype(mx.float32),
        scale=128**-0.5,
        mask=mask,
    ).astype(mx.bfloat16)
    assert actual.shape == query.shape
    assert actual.dtype == mx.bfloat16
    assert_allclose(actual, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


def test_online_decode_fully_masked_partition_is_finite_zero():
    query = _fixture((1, 4, 1, 128), 0.1)
    key = _fixture((1, 1, 257, 128), 0.7)
    value = _fixture(key.shape, 1.3)
    actual = decode_attention_online(
        query, key, value, 128**-0.5, mx.full((257,), -mx.inf)
    )
    mx.eval(actual)
    assert bool(mx.all(mx.isfinite(actual)).item())
    assert float(mx.max(mx.abs(actual)).item()) == 0.0


@pytest.mark.parametrize("heads", (8, 32))
@pytest.mark.parametrize("already_evaluated", (False, True))
def test_qk_norm_rope_matches_composition_before_and_after_eval(
    heads, already_evaluated
):
    """A materialized RMSNorm output must keep the unfused fallback correct."""
    x = _fixture((1, 1, heads, 128), 0.1)
    weight = _fixture((128,), 0.7)
    norm = FastRMSNorm(128, weight, eps=1e-5)
    rope = FastRoPE(128, 32768, 1000000)
    normalized = norm(x)
    if already_evaluated:
        mx.eval(normalized)
    actual = _qk_norm_rope4(normalized, rope, offset=8192)
    expected = rope(normalized, offset=8192)
    assert_allclose(actual, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("rows,outputs", ((32, 64), (33, 65)))
def test_quantized_prefill_full_and_partial_tiles_match_mlx(rows, outputs):
    x = _fixture((rows, 256), 0.1)
    weight = _fixture((outputs, 256), 0.7)
    packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
    actual = quantized_matmul(
        scales, biases, 128, 4, x, packed, transpose_b=True, use_simdgroup=True
    )
    expected = mx.quantized_matmul(
        x, packed, scales, biases, group_size=128, bits=4, transpose=True
    )
    assert actual.shape == (rows, outputs)
    assert_allclose(actual, expected, mx.bfloat16, atol=0.25, rtol=1e-2)


def test_selected_default_and_teaching_checkpoint_keep_cached_decode_outputs():
    fixture = tiny_qwen3_mlx_model(head_dim=128, seed=0)
    default = Qwen3ModelWeek2(fixture)
    teaching = Qwen3ModelWeek2(fixture, checkpoint="tiled-prefill")
    actual_cache = default.create_kv_cache(capacity=13)
    expected_cache = teaching.create_kv_cache(capacity=13)
    tokens = mx.array([list(range(1, 11))], dtype=mx.int32)
    for offset in (0, 10, 11, 12):
        actual = default(tokens, offset, actual_cache, logits_to_keep=1)
        expected = teaching(tokens, offset, expected_cache, logits_to_keep=1)
        assert_allclose(actual, expected, mx.bfloat16, atol=0.75, rtol=0.05)
        tokens = mx.array([[13]], dtype=mx.int32)
