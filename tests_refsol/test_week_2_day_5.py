"""Week 2 Day 5 public tiled-attention and selected-engine tests."""

from math import prod

import mlx.core as mx
import pytest

from .tiny_llm_base import Qwen3ModelWeek2, dense_prefill_attention_mma
from .utils import assert_allclose, tiny_qwen3_mlx_model


def _fixture(shape: tuple[int, ...], phase: float) -> mx.array:
    values = mx.sin(mx.arange(prod(shape), dtype=mx.float32) * 0.017 + phase)
    return values.reshape(shape).astype(mx.bfloat16)


def _model_fixture(seed: int):
    state = mx.random.state[:]
    try:
        mx.random.seed(seed)
        return tiny_qwen3_mlx_model(head_dim=128)
    finally:
        mx.random.state[:] = state


def _run(model, length: int):
    tokens = mx.array([list(range(1, length + 1))], dtype=mx.int32)
    output = model(tokens, 0, model.create_kv_cache(capacity=length))
    mx.eval(output)
    assert output.dtype == mx.bfloat16
    assert output.shape[:2] == (1, length)
    return output


def test_task_1_tiled_prefill_matches_mlx_causal_gqa():
    query = _fixture((1, 4, 33, 128), 0.1)
    key = _fixture((1, 2, 47, 128), 0.7)
    value = _fixture(key.shape, 1.3)
    actual = dense_prefill_attention_mma(query, key, value, 128**-0.5, "causal")
    expected = mx.fast.scaled_dot_product_attention(
        query.astype(mx.float32),
        key.astype(mx.float32),
        value.astype(mx.float32),
        scale=128**-0.5,
        mask="causal",
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


@pytest.mark.parametrize("length", (3, 10))
def test_task_3_tiled_checkpoint_matches_readable_model(length: int):
    fixture = _model_fixture(4)
    tiled = Qwen3ModelWeek2(fixture, checkpoint="tiled-prefill")
    readable = Qwen3ModelWeek2(fixture, checkpoint="swiglu")
    actual = _run(tiled, length)
    expected = _run(readable, length)
    assert_allclose(actual, expected, mx.bfloat16, atol=0.75, rtol=0.05)


@pytest.mark.parametrize(
    "disabled",
    (
        {"use_bounded_kv_capacity": False},
        {"use_register_cached_rms_norm": False},
        {"use_tiled_prefill_attention": False},
    ),
)
def test_selected_controls_can_be_disabled_independently(disabled):
    fixture = _model_fixture(0)
    selected = Qwen3ModelWeek2(fixture, checkpoint="selected")
    switched = Qwen3ModelWeek2(fixture, checkpoint="selected", **disabled)
    actual = _run(selected, 10)
    expected = _run(switched, 10)
    assert_allclose(actual, expected, mx.bfloat16, atol=0.75, rtol=0.05)


def test_selected_is_default_and_matches_explicit_checkpoint():
    fixture = _model_fixture(0)
    default = Qwen3ModelWeek2(fixture)
    explicit = Qwen3ModelWeek2(fixture, checkpoint="selected")
    assert_allclose(_run(default, 10), _run(explicit, 10), mx.bfloat16)


@pytest.mark.parametrize("checkpoint", ("decode-attention", "split-k"))
def test_retired_experiments_are_not_week2_checkpoints(checkpoint):
    with pytest.raises(ValueError, match="unknown Week 2 checkpoint"):
        Qwen3ModelWeek2(_model_fixture(0), checkpoint=checkpoint)
