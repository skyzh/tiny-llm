"""Week 2 Day 2 request-bounded dense KV-cache tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import Qwen3ModelWeek2, TinyKvFullCache
from .utils import assert_allclose, tiny_qwen3_mlx_model


def _chunk(start: int, length: int):
    key = mx.arange(start, start + length * 2, dtype=mx.float32).reshape(
        1, 1, length, 2
    )
    return key, key + 100


def test_task_1_capacity_cache_exposes_only_the_logical_prefix():
    cache = TinyKvFullCache(capacity=5)
    key_1, value_1 = _chunk(0, 2)
    key_2, value_2 = _chunk(4, 1)

    cache.update_and_fetch(key_1, value_1)
    cached_key, cached_value, offset, _ = cache.update_and_fetch(key_2, value_2)
    mx.eval(cached_key, cached_value)

    assert offset == 3
    assert cached_key.shape == cached_value.shape == (1, 1, 3, 2)
    assert cache.key_values[0].shape == cache.key_values[1].shape == (1, 1, 5, 2)
    assert_allclose(cached_key, mx.concat([key_1, key_2], axis=2), mx.float32)
    assert_allclose(cached_value, mx.concat([value_1, value_2], axis=2), mx.float32)
    assert cache.logical_copy_bytes == 0
    assert cache.physical_growth_copy_bytes == 0
    assert cache.slice_write_bytes == (
        key_1.nbytes + value_1.nbytes + key_2.nbytes + value_2.nbytes
    )


def test_task_2_rewind_reuses_capacity_and_overflow_is_transactional():
    cache = TinyKvFullCache(capacity=3)
    key, value = _chunk(0, 2)
    cache.update_and_fetch(key, value)
    cache.rewind(1)
    replacement_key, replacement_value = _chunk(20, 2)
    cached_key, cached_value, offset, _ = cache.update_and_fetch(
        replacement_key, replacement_value
    )
    mx.eval(cached_key, cached_value)

    assert offset == 3
    assert cache.physical_growth_copy_bytes == 0
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

    before = tuple(array.tolist() for array in cache.key_values)
    with pytest.raises(ValueError, match="capacity 3 exceeded"):
        extra_key, extra_value = _chunk(30, 1)
        cache.update_and_fetch(extra_key, extra_value)
    mx.eval(*cache.key_values)
    assert tuple(array.tolist() for array in cache.key_values) == before
    assert cache.offset == 3

    cache.reset()
    assert cache.offset == 0
    assert cache.key_values is not None


def test_task_3_capacity_checkpoint_runs_the_week2_engine():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="capacity-cache")
    assert model.use_bounded_kv_capacity

    bounded = model.create_kv_cache(capacity=3)
    readable = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(), checkpoint="kv-cache"
    ).create_kv_cache()
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    actual = model(inputs, 0, bounded)
    expected = model(inputs, 0, readable)

    assert_allclose(actual, expected, mx.bfloat16)
    assert all(cache.capacity == 3 for cache in bounded)
    assert all(cache.slice_write_bytes > 0 for cache in bounded)
