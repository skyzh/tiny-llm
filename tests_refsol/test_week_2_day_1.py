"""Week 2 Day 1 dense KV-cache tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import Embedding, Qwen3ModelWeek2, RMSNorm, RoPE, TinyKvFullCache
from .utils import assert_allclose, tiny_qwen3_mlx_model


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
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="kv-cache")
    layer = model.layers_inner[0]

    assert isinstance(model.embedding, Embedding)
    assert isinstance(layer.input_layernorm, RMSNorm)
    assert isinstance(layer.self_attn.rope, RoPE)
    assert not model.use_bounded_kv_capacity
    assert not layer.self_attn.use_decode_attention
    assert not layer.mlp.use_fast_swiglu
    assert len(model.create_kv_cache()) == model.num_hidden_layers

    output = model(mx.array([[1, 2]], dtype=mx.int32), 0, model.create_kv_cache())
    assert output.dtype == mx.bfloat16


def test_task_3_rejects_a_position_that_disagrees_with_the_cache():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="kv-cache")
    with pytest.raises(ValueError):
        model(mx.array([[1]], dtype=mx.int32), 1, model.create_kv_cache())


# Second checkpoint: request-bounded capacity.


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
    assert cache.key_values[0].shape == cache.key_values[1].shape == (1, 1, 5, 2)
    assert_allclose(cached_key, mx.concat([key_1, key_2], axis=2), mx.float32)
    assert_allclose(cached_value, mx.concat([value_1, value_2], axis=2), mx.float32)
    assert cache.logical_copy_bytes == 0
    assert cache.physical_growth_copy_bytes == 0
    assert cache.slice_write_bytes == (
        key_1.nbytes + value_1.nbytes + key_2.nbytes + value_2.nbytes
    )


def test_capacity_rewind_reuses_storage_and_overflow_is_transactional():
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

    movement_counters = (
        cache.logical_copy_bytes,
        cache.physical_growth_copy_bytes,
        cache.slice_write_bytes,
        cache.growth_copy_bytes,
    )
    cache.reset()
    assert cache.offset == 0
    assert cache.key_values is not None
    assert (
        cache.logical_copy_bytes,
        cache.physical_growth_copy_bytes,
        cache.slice_write_bytes,
        cache.growth_copy_bytes,
    ) == movement_counters


def test_capacity_checkpoint_runs_the_week2_engine():
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
