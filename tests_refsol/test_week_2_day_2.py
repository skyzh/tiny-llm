"""Week 2 Day 2 dense KV-cache tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import TinyKvFullCache
from .utils import assert_allclose


def test_task_1_full_cache_appends_chunks():
    cache = TinyKvFullCache()
    key_1 = mx.random.normal((1, 2, 3, 4)).astype(mx.float32)
    value_1 = mx.random.normal((1, 2, 3, 4)).astype(mx.float32)
    key_2 = mx.random.normal((1, 2, 2, 4)).astype(mx.float32)
    value_2 = mx.random.normal((1, 2, 2, 4)).astype(mx.float32)

    cached_key, cached_value, offset, mask = cache.update_and_fetch(
        key_1, value_1, mask="causal"
    )
    assert offset == 3
    assert mask == "causal"
    assert_allclose(cached_key, key_1, mx.float32)
    assert_allclose(cached_value, value_1, mx.float32)

    cached_key, cached_value, offset, _ = cache.update_and_fetch(key_2, value_2)
    assert offset == 5
    assert_allclose(cached_key, mx.concat([key_1, key_2], axis=2), mx.float32)
    assert_allclose(cached_value, mx.concat([value_1, value_2], axis=2), mx.float32)


def test_task_1_full_cache_rewind():
    cache = TinyKvFullCache()
    key = mx.random.normal((1, 2, 5, 4)).astype(mx.float32)
    value = mx.random.normal((1, 2, 5, 4)).astype(mx.float32)
    cache.update_and_fetch(key, value)
    cache.rewind(2)
    assert cache.offset == 3
    assert_allclose(cache.key_values[0], key[:, :, :3], mx.float32)
    assert_allclose(cache.key_values[1], value[:, :, :3], mx.float32)


def test_task_1_dense_cache_has_no_paged_metadata():
    cache = TinyKvFullCache()
    key = mx.zeros((1, 1, 1, 4))
    value = mx.zeros((1, 1, 1, 4))
    with pytest.raises(NotImplementedError):
        cache.update_and_fetch_paged(key, value)
