from copy import copy
from math import prod

from extensions_ref.tiny_llm_ext_ref import _ext
import mlx.core as mx
import pytest

from tiny_llm_ref.kv_cache import TinyKvFullCache
from tiny_llm_ref.qwen3_week2 import Qwen3MultiHeadAttention


def bf16_fixture(shape, phase):
    return (
        mx.sin(mx.arange(prod(shape), dtype=mx.float32) * 0.017 + phase)
        .reshape(shape)
        .astype(mx.bfloat16)
    )


def assert_bits_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == mx.bfloat16
    assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()


@pytest.mark.parametrize("context", (4092, 4096, 4095))
def test_readable_singleton_preserves_projection_and_cache(context, monkeypatch):
    attention = Qwen3MultiHeadAttention(
        hidden_size=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        wq=bf16_fixture((4096, 32), 0.1),
        wk=bf16_fixture((1024, 32), 0.2),
        wv=bf16_fixture((1024, 32), 0.3),
        wo=bf16_fixture((32, 4096), 0.4),
        q_norm=mx.ones(128, dtype=mx.bfloat16),
        k_norm=mx.ones(128, dtype=mx.bfloat16),
        max_seq_len=4096,
        use_fast_rms_norm=False,
        use_fast_rope=False,
    )
    cache = TinyKvFullCache(capacity=4096)
    history_shape = (1, 8, context - 1, 128)
    old_key, old_value, _, _ = cache.update_and_fetch(
        bf16_fixture(history_shape, 0.1), bf16_fixture(history_shape, 0.9)
    )
    mx.eval(old_key, old_value, *cache.key_values)
    old_key_snapshot, old_value_snapshot = mx.contiguous(old_key), mx.contiguous(old_value)
    mx.eval(old_key_snapshot, old_value_snapshot)
    control_cache = copy(cache)
    token = bf16_fixture((1, 1, 32), 0.6)
    mask = mx.where(mx.arange(context) % 13 == 0, -mx.inf, 0).astype(mx.float32)

    with monkeypatch.context() as patch:
        patch.setattr(_ext, "_try_week2_bf16_history_scores", None)
        control = attention(token, context - 1, control_cache, mask)
        mx.eval(control, *control_cache.key_values)

    score = _ext._try_week2_bf16_history_scores
    calls = []

    def observe(*args):
        calls.append(args[1].shape[-2])
        return score(*args)

    monkeypatch.setattr(_ext, "_try_week2_bf16_history_scores", observe)
    actual = attention(token, context - 1, cache, mask)
    mx.eval(actual, *cache.key_values)
    assert calls == ([context] if context in (4092, 4096) else [])
    assert_bits_equal(actual, control)
    assert cache.offset == control_cache.offset == context
    assert cache.slice_write_bytes == control_cache.slice_write_bytes
    assert cache.growth_copy_bytes == control_cache.growth_copy_bytes
    for a, b in zip(cache.key_values, control_cache.key_values, strict=True):
        assert_bits_equal(a, b)
    assert_bits_equal(old_key, old_key_snapshot)
    assert_bits_equal(old_value, old_value_snapshot)


def test_bf16_guard_checks_offsets_for_both_histories():
    query = bf16_fixture((1, 32, 1, 128), 0.1)
    backing = bf16_fixture((1, 8, 4093, 128), 0.2)
    mx.eval(query, backing)
    valid = backing[:, :, :4092, :]
    shifted = backing[:, :, 1:, :]
    try_scores = _ext._try_week2_bf16_history_scores
    assert try_scores(query, valid, shifted) is None
    assert try_scores(query, shifted, valid) is None
