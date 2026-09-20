"""Week 2 Day 1 dense KV-cache tests."""

import ctypes

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

    cached_key, cached_value, offset, mask = cache.update_and_fetch(
        key_1, value_1, mask="causal"
    )
    assert offset == 3
    assert mask == "causal"
    assert_allclose(cached_key, key_1, mx.bfloat16)
    assert_allclose(cached_value, value_1, mx.bfloat16)

    cached_key, cached_value, offset, _ = cache.update_and_fetch(key_2, value_2)
    assert offset == 5
    assert_allclose(cached_key, mx.concat([key_1, key_2], axis=2), mx.bfloat16)
    assert_allclose(cached_value, mx.concat([value_1, value_2], axis=2), mx.bfloat16)


def _storage_address(array: mx.array) -> int:
    """Return the address exported by the evaluated MLX storage buffer."""

    mx.eval(array)
    return ctypes.addressof(ctypes.c_char.from_buffer(memoryview(array)))


def test_task_1_full_cache_appends_in_place_until_capacity_is_exhausted():
    cache = TinyKvFullCache()
    first_key = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.bfloat16)
    first_value = (first_key + 100).astype(mx.bfloat16)
    cache.update_and_fetch(first_key, first_value)

    assert cache.capacity == 4
    key_address = _storage_address(cache._key_storage)
    value_address = _storage_address(cache._value_storage)
    copied_before = cache.growth_copy_bytes

    second_key = mx.full((1, 2, 1, 4), 31, dtype=mx.bfloat16)
    second_value = mx.full((1, 2, 1, 4), 41, dtype=mx.bfloat16)
    cached_key, cached_value, offset, _ = cache.update_and_fetch(
        second_key, second_value
    )

    assert offset == cache.capacity == 4
    assert _storage_address(cache._key_storage) == key_address
    assert _storage_address(cache._value_storage) == value_address
    assert cache.growth_copy_bytes == copied_before == 0
    assert_allclose(cached_key, mx.concat([first_key, second_key], axis=2), mx.bfloat16)
    assert_allclose(
        cached_value, mx.concat([first_value, second_value], axis=2), mx.bfloat16
    )


def test_task_1_full_cache_growth_relocates_only_after_capacity_is_exhausted():
    cache = TinyKvFullCache()
    first_key = mx.arange(32).reshape(1, 2, 4, 4).astype(mx.bfloat16)
    first_value = (first_key + 100).astype(mx.bfloat16)
    cache.update_and_fetch(first_key, first_value)
    key_address = _storage_address(cache._key_storage)
    value_address = _storage_address(cache._value_storage)

    next_key = mx.full((1, 2, 1, 4), 51, dtype=mx.bfloat16)
    next_value = mx.full((1, 2, 1, 4), 61, dtype=mx.bfloat16)
    cached_key, cached_value, offset, _ = cache.update_and_fetch(next_key, next_value)

    assert offset == 5
    assert cache.capacity == 8
    assert _storage_address(cache._key_storage) != key_address
    assert _storage_address(cache._value_storage) != value_address
    assert cache.growth_copy_bytes == first_key.nbytes + first_value.nbytes
    assert_allclose(cached_key, mx.concat([first_key, next_key], axis=2), mx.bfloat16)
    assert_allclose(
        cached_value, mx.concat([first_value, next_value], axis=2), mx.bfloat16
    )


def test_tasks_2_and_3_cached_checkpoint_is_runnable_and_readable():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="kv-cache")
    layer = model.layers_inner[0]

    assert isinstance(model.embedding, Embedding)
    assert isinstance(layer.input_layernorm, RMSNorm)
    assert isinstance(layer.self_attn.rope, RoPE)
    assert not layer.self_attn.use_decode_attention
    assert not layer.mlp.use_fast_swiglu
    assert len(model.create_kv_cache()) == model.num_hidden_layers

    output = model(mx.array([[1, 2]], dtype=mx.int32), 0, model.create_kv_cache())
    assert output.dtype == mx.bfloat16


def test_task_3_rejects_a_position_that_disagrees_with_the_cache():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="kv-cache")
    with pytest.raises(ValueError):
        model(mx.array([[1]], dtype=mx.int32), 1, model.create_kv_cache())
