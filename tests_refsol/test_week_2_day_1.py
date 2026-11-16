"""Week 2 Day 1 dense KV-cache tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import Qwen3ModelWeek2, TinyKvFullCache
from .utils import assert_allclose, tiny_qwen3_mlx_model


def _fixed_fixture(seed: int):
    return tiny_qwen3_mlx_model(seed=seed)


def test_task_1_full_cache_appends_chunks():
    cache = TinyKvFullCache()
    key_1 = mx.random.normal((1, 2, 3, 4)).astype(mx.bfloat16)
    value_1 = mx.random.normal((1, 2, 3, 4)).astype(mx.bfloat16)
    key_2 = mx.random.normal((1, 2, 2, 4)).astype(mx.bfloat16)
    value_2 = mx.random.normal((1, 2, 2, 4)).astype(mx.bfloat16)

    first_update = cache.update_and_fetch(key_1, value_1, mask="causal")
    assert first_update is not None, (
        "implement the TinyKvFullCache.update_and_fetch learner seam"
    )
    cached_key, cached_value, offset, mask = first_update
    assert offset == 3
    assert mask == "causal"
    assert_allclose(cached_key, key_1, mx.bfloat16)
    assert_allclose(cached_value, value_1, mx.bfloat16)

    cached_key, cached_value, offset, _ = cache.update_and_fetch(key_2, value_2)
    assert offset == 5
    assert_allclose(cached_key, mx.concat([key_1, key_2], axis=2), mx.bfloat16)
    assert_allclose(cached_value, mx.concat([value_1, value_2], axis=2), mx.bfloat16)


def test_tasks_2_and_3_cached_checkpoint_is_runnable_and_readable():
    fixture = _fixed_fixture(0)
    model = Qwen3ModelWeek2(fixture, checkpoint="kv-cache")
    cache = model.create_kv_cache()
    assert len(cache) == fixture.args.num_hidden_layers

    prefill = model(mx.array([[1, 2]], dtype=mx.int32), 0, cache)
    decoded = model(mx.array([[3]], dtype=mx.int32), 2, cache)
    complete = model(mx.array([[1, 2, 3]], dtype=mx.int32), 0, model.create_kv_cache())
    assert prefill.dtype == decoded.dtype == complete.dtype == mx.bfloat16
    assert prefill.shape == (1, 2, fixture.args.vocab_size)
    assert decoded.shape == (1, 1, fixture.args.vocab_size)
    assert_allclose(decoded, complete[:, -1:, :], mx.bfloat16)


def test_task_3_rejects_a_position_that_disagrees_with_the_cache():
    model = Qwen3ModelWeek2(_fixed_fixture(0), checkpoint="kv-cache")
    with pytest.raises(ValueError):
        model(mx.array([[1]], dtype=mx.int32), 1, model.create_kv_cache())


# One cache checkpoint; explicit bounds remain optional.


def _chunk(start: int, length: int):
    key = mx.arange(start, start + length * 2, dtype=mx.float32).reshape(
        1, 1, length, 2
    )
    return key, key + 100


def test_capacity_cache_exposes_only_the_logical_prefix():
    cache = TinyKvFullCache(capacity=5)
    key_1, value_1 = _chunk(0, 2)
    key_2, value_2 = _chunk(4, 1)

    cache.update_and_fetch(key_1, value_1)
    prefix_update = cache.update_and_fetch(key_2, value_2)
    assert prefix_update is not None, (
        "implement the TinyKvFullCache capacity-cache update_and_fetch learner seam"
    )
    cached_key, cached_value, offset, _ = prefix_update
    mx.eval(cached_key, cached_value)

    assert offset == 3
    assert cached_key.shape == cached_value.shape == (1, 1, 3, 2)
    assert_allclose(cached_key, mx.concat([key_1, key_2], axis=2), mx.float32)
    assert_allclose(cached_value, mx.concat([value_1, value_2], axis=2), mx.float32)

    # Unused request capacity must never appear in the returned K/V prefix.
    assert cached_key.shape[2] == 3 < cache.capacity


def test_capacity_rewind_and_overflow_preserve_the_logical_prefix():
    cache = TinyKvFullCache(capacity=3)
    key, value = _chunk(0, 2)
    cache.update_and_fetch(key, value)
    cache.rewind(1)
    replacement_key, replacement_value = _chunk(20, 2)
    replacement_update = cache.update_and_fetch(replacement_key, replacement_value)
    assert replacement_update is not None, (
        "implement the TinyKvFullCache capacity-cache update_and_fetch learner seam"
    )
    cached_key, cached_value, offset, _ = replacement_update
    mx.eval(cached_key, cached_value)

    assert offset == 3
    assert_allclose(
        cached_key,
        mx.concat([key[:, :, :1], replacement_key], axis=2),
        mx.float32,
    )
    assert_allclose(
        cached_value,
        mx.concat([value[:, :, :1], replacement_value], axis=2),
        mx.float32,
    )

    with pytest.raises(ValueError, match="capacity 3 exceeded"):
        extra_key, extra_value = _chunk(30, 1)
        cache.update_and_fetch(extra_key, extra_value)
    assert cache.offset == 3

    # A rejected append must leave the retained prefix available for rewind.
    cache.rewind(1)
    final_key, final_value = _chunk(40, 1)
    cached_key, cached_value, offset, _ = cache.update_and_fetch(final_key, final_value)
    assert offset == 3
    assert_allclose(
        cached_key,
        mx.concat([key[:, :, :1], replacement_key[:, :, :1], final_key], axis=2),
        mx.float32,
    )
    assert_allclose(
        cached_value,
        mx.concat([value[:, :, :1], replacement_value[:, :, :1], final_value], axis=2),
        mx.float32,
    )

    cache.reset()
    assert cache.offset == 0
    restarted_key, restarted_value = _chunk(60, 1)
    cached_key, cached_value, offset, _ = cache.update_and_fetch(
        restarted_key, restarted_value
    )
    assert offset == 1
    assert_allclose(cached_key, restarted_key, mx.float32)
    assert_allclose(cached_value, restarted_value, mx.float32)


def test_capacity_checkpoint_runs_the_week2_engine():
    fixture = _fixed_fixture(0)
    model = Qwen3ModelWeek2(fixture, checkpoint="kv-cache")
    readable_model = Qwen3ModelWeek2(fixture, checkpoint="kv-cache")
    bounded = model.create_kv_cache(capacity=3)
    readable = readable_model.create_kv_cache()
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    actual = model(inputs, 0, bounded)
    expected = readable_model(inputs, 0, readable)

    assert actual.dtype == expected.dtype == mx.bfloat16
    assert actual.shape == expected.shape == (1, 3, fixture.args.vocab_size)
    assert_allclose(actual, expected, mx.bfloat16)
    assert all(cache.offset == 3 for cache in bounded)
    with pytest.raises(ValueError, match="capacity"):
        model(mx.array([[4]], dtype=mx.int32), 3, bounded)


@pytest.mark.parametrize("length", (255, 256, 257, 513))
def test_chunk_growth_returns_only_the_initialized_prefix(length):
    cache = TinyKvFullCache()
    key, value = _chunk(0, length)
    result = cache.update_and_fetch(key, value)
    assert result is not None, "implement chunked TinyKvFullCache storage"
    actual_key, actual_value, offset, _ = result
    assert offset == length
    assert cache.allocated_capacity == ((length + 255) // 256) * 256
    assert actual_key.shape[2] == actual_value.shape[2] == length
    assert_allclose(actual_key, key, mx.float32)
    assert_allclose(actual_value, value, mx.float32)


def test_chunk_growth_rewind_and_reset_keep_storage_and_hide_abandoned_tokens():
    cache = TinyKvFullCache(growth_step=4)
    key, value = _chunk(0, 3)
    cache.update_and_fetch(key, value)
    cache.rewind(2)
    replacement, replacement_value = _chunk(30, 4)
    keys, values, offset, _ = cache.update_and_fetch(replacement, replacement_value)
    assert offset == 5 and cache.allocated_capacity == 8
    assert_allclose(keys, mx.concat([key[:, :, :1], replacement], axis=2), mx.float32)
    assert_allclose(
        values, mx.concat([value[:, :, :1], replacement_value], axis=2), mx.float32
    )
    cache.reset()
    assert cache.offset == 0 and cache.allocated_capacity == 8
    keys, values, offset, _ = cache.update_and_fetch(key[:, :, :1], value[:, :, :1])
    assert offset == 1 and keys.shape[2] == values.shape[2] == 1


def test_capacity_cache_command_is_only_a_compatibility_alias():
    fixture = _fixed_fixture(1)
    legacy = Qwen3ModelWeek2(fixture, checkpoint="capacity-cache")
    current = Qwen3ModelWeek2(fixture, checkpoint="kv-cache")
    assert legacy.checkpoint == current.checkpoint == "kv-cache"
    assert not legacy.use_bounded_kv_capacity and not current.use_bounded_kv_capacity
