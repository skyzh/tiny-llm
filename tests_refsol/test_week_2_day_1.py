"""Week 2 Day 1 dense KV-cache tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import Embedding, Qwen3ModelWeek2, RMSNorm, RoPE, TinyKvFullCache
from .utils import assert_allclose, tiny_qwen3_mlx_model


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


def test_tasks_2_and_3_cached_checkpoint_is_runnable_and_readable():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="kv-cache")
    layer = model.layers_inner[0]

    assert isinstance(model.embedding, Embedding)
    assert isinstance(layer.input_layernorm, RMSNorm)
    assert isinstance(layer.self_attn.rope, RoPE)
    assert not layer.self_attn.use_decode_attention
    assert not layer.mlp.use_fast_swiglu
    assert len(model.create_kv_cache()) == model.num_hidden_layers


def test_task_3_rejects_a_position_that_disagrees_with_the_cache():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="kv-cache")
    with pytest.raises(ValueError, match="does not match model offset"):
        model(mx.array([[1]], dtype=mx.int32), 1, model.create_kv_cache())
