"""Focused request-bounded dense KV capacity controls."""

import mlx.core as mx
import pytest

from tiny_llm_ref.generate import simple_generate_with_kv_cache
from tiny_llm_ref.kv_cache import TinyKvFullCache
from tiny_llm_ref.qwen3_week2 import Qwen3ModelWeek2
from .utils import assert_allclose, tiny_qwen3_mlx_model


def _chunk(start: int, length: int, *, dtype=mx.float32):
    values = mx.arange(start, start + length * 2, dtype=dtype).reshape(1, 1, length, 2)
    return values, values + 100


def test_capacity_cache_writes_only_new_slices_and_hides_the_tail():
    cache = TinyKvFullCache(capacity=5)
    key_1, value_1 = _chunk(0, 2)
    key_2, value_2 = _chunk(4, 1)

    first_key, first_value, offset, _ = cache.update_and_fetch(key_1, value_1)
    assert offset == 2
    assert first_key.shape == first_value.shape == (1, 1, 2, 2)

    cached_key, cached_value, offset, _ = cache.update_and_fetch(key_2, value_2)
    mx.eval(cached_key, cached_value)

    assert offset == 3
    assert cached_key.shape == cached_value.shape == (1, 1, 3, 2)
    assert cache.key_values[0].shape == cache.key_values[1].shape == (1, 1, 5, 2)
    assert_allclose(cached_key, mx.concat([key_1, key_2], axis=2), mx.float32)
    assert_allclose(cached_value, mx.concat([value_1, value_2], axis=2), mx.float32)
    assert cache.key_values[0][:, :, 3:].tolist() == [[[[0.0, 0.0], [0.0, 0.0]]]]
    assert cache.key_values[1][:, :, 3:].tolist() == [[[[0.0, 0.0], [0.0, 0.0]]]]
    assert cache.logical_copy_bytes == 0
    assert cache.physical_growth_copy_bytes == 0
    assert cache.growth_copy_bytes == 0
    assert (
        cache.slice_write_bytes
        == key_1.nbytes + value_1.nbytes + key_2.nbytes + value_2.nbytes
    )


def test_disabled_control_keeps_concatenate_accounting():
    cache = TinyKvFullCache()
    key_1, value_1 = _chunk(0, 2)
    key_2, value_2 = _chunk(4, 1)

    cache.update_and_fetch(key_1, value_1)
    cached_key, cached_value, offset, _ = cache.update_and_fetch(key_2, value_2)

    copied = key_1.nbytes + value_1.nbytes
    assert offset == 3
    assert_allclose(cached_key, mx.concat([key_1, key_2], axis=2), mx.float32)
    assert_allclose(cached_value, mx.concat([value_1, value_2], axis=2), mx.float32)
    assert cache.capacity is None
    assert cache.logical_copy_bytes == copied
    assert cache.physical_growth_copy_bytes == copied
    assert cache.growth_copy_bytes == copied
    assert cache.slice_write_bytes == 0


def test_capacity_boundary_is_transactional_and_zero_append_is_a_noop():
    for invalid_capacity in (-1, 1.5, True):
        with pytest.raises(ValueError, match="non-negative integer"):
            TinyKvFullCache(capacity=invalid_capacity)

    cache = TinyKvFullCache(capacity=2)
    key, value = _chunk(0, 2)
    cache.update_and_fetch(key, value)
    mx.eval(*cache.key_values)
    before = tuple(array.tolist() for array in cache.key_values)
    before_counters = (cache.offset, cache.slice_write_bytes)

    empty_key, empty_value = _chunk(0, 0)
    cached_key, cached_value, offset, _ = cache.update_and_fetch(empty_key, empty_value)
    assert cached_key.shape == cached_value.shape == (1, 1, 2, 2)
    assert offset == 2
    assert (cache.offset, cache.slice_write_bytes) == before_counters

    extra_key, extra_value = _chunk(4, 1)
    with pytest.raises(ValueError, match="capacity 2 exceeded"):
        cache.update_and_fetch(extra_key, extra_value)

    mx.eval(*cache.key_values)
    assert tuple(array.tolist() for array in cache.key_values) == before
    assert (cache.offset, cache.slice_write_bytes) == before_counters


def test_capacity_rewind_reuses_storage_and_overwrites_the_logical_suffix():
    cache = TinyKvFullCache(capacity=4)
    first_key, first_value = _chunk(0, 3)
    cache.update_and_fetch(first_key, first_value)
    storage_before_rewind = cache.key_values

    cache.rewind(2)
    assert all(
        current is original
        for current, original in zip(cache.key_values, storage_before_rewind)
    )
    replacement_key, replacement_value = _chunk(20, 2)
    cached_key, cached_value, offset, _ = cache.update_and_fetch(
        replacement_key, replacement_value
    )
    mx.eval(cached_key, cached_value)

    assert offset == 3
    assert cache.key_values[0].shape == cache.key_values[1].shape == (1, 1, 4, 2)
    assert_allclose(
        cached_key,
        mx.concat([first_key[:, :, :1], replacement_key], axis=2),
        mx.float32,
    )
    assert_allclose(
        cached_value,
        mx.concat([first_value[:, :, :1], replacement_value], axis=2),
        mx.float32,
    )

    cache.rewind(3)
    assert cache.offset == 0
    assert cache.key_values is not None


def test_capacity_mode_matches_the_legacy_model_cache():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="kv-cache")
    bounded = model.create_kv_cache(capacity=3)
    legacy = model.create_kv_cache()

    for tokens, offset in (
        (mx.array([[1, 2]], dtype=mx.int32), 0),
        (mx.array([[3]], dtype=mx.int32), 2),
    ):
        bounded_output = model(tokens, offset, bounded)
        legacy_output = model(tokens, offset, legacy)
        assert_allclose(bounded_output, legacy_output, mx.bfloat16)

    for bounded_layer, legacy_layer in zip(bounded, legacy):
        bounded_keys, bounded_values = bounded_layer._logical_key_values()
        legacy_keys, legacy_values = legacy_layer.key_values
        assert_allclose(bounded_keys, legacy_keys, mx.bfloat16)
        assert_allclose(bounded_values, legacy_values, mx.bfloat16)
        assert bounded_layer.physical_growth_copy_bytes == 0
        assert legacy_layer.physical_growth_copy_bytes > 0


def test_week2_factory_and_generation_bind_the_request_capacity(capsys):
    model = object.__new__(Qwen3ModelWeek2)
    model.num_hidden_layers = 2
    caches = model.create_kv_cache(capacity=7)
    assert [cache.capacity for cache in caches] == [7, 7]

    class Detokenizer:
        last_segment = ""

        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = ""

        def finalize(self):
            self.last_segment = ""

    class Tokenizer:
        eos_token_id = 99
        detokenizer = Detokenizer()

        def encode(self, prompt, add_special_tokens=False):
            assert prompt == "hi"
            assert add_special_tokens is False
            return [4, 5]

    class Model:
        def __init__(self):
            self.capacities = []
            self.caches = []

        def create_kv_cache(self, capacity=None):
            self.capacities.append(capacity)
            cache = [TinyKvFullCache(capacity=capacity)]
            self.caches.append(cache)
            return cache

        def __call__(self, inputs, offset, cache, logits_to_keep=1):
            assert offset == cache[0].offset
            values = mx.ones((1, 1, inputs.shape[1], 1), dtype=mx.float32)
            cache[0].update_and_fetch(values, values)
            return mx.array([[[0.0, 1.0, 0.0]]], dtype=mx.float32)

    enabled = Model()
    simple_generate_with_kv_cache(
        enabled,
        Tokenizer(),
        "hi",
        max_tokens=2,
        use_bounded_kv_capacity=True,
    )
    assert enabled.capacities == [4]
    assert enabled.caches[0][0].offset == 3
    assert enabled.caches[0][0].slice_write_bytes > 0

    disabled = Model()
    simple_generate_with_kv_cache(disabled, Tokenizer(), "hi", max_tokens=2)
    assert disabled.capacities == [None]
    assert disabled.caches[0][0].offset == 3
    assert disabled.caches[0][0].logical_copy_bytes > 0
    enabled_keys, enabled_values = enabled.caches[0][0].key_values
    disabled_keys, disabled_values = disabled.caches[0][0].key_values
    assert_allclose(enabled_keys[:, :, :3], disabled_keys, mx.float32)
    assert_allclose(enabled_values[:, :, :3], disabled_values, mx.float32)
    assert capsys.readouterr().out == ""
