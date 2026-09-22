"""Week 2 Day 7 measured cumulative-selection tests."""

import mlx.core as mx

from .tiny_llm_base import FastRMSNorm, Qwen3ModelWeek2
from .utils import assert_allclose, tiny_qwen3_mlx_model


def test_selected_checkpoint_contains_exactly_the_three_selected_mechanisms():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="selected")
    layer = model.layers_inner[0]

    assert model.use_bounded_kv_capacity
    assert isinstance(layer.input_layernorm, FastRMSNorm)
    assert layer.self_attn.use_tiled_prefill_attention
    assert layer.self_attn.wq.use_simdgroup_matmul
    assert not layer.self_attn.wq.use_split_k_matmul
    assert not hasattr(layer.self_attn, "use_decode_attention")

    cache = model.create_kv_cache(capacity=4)
    output = model(mx.array([[1, 2, 3, 4]], dtype=mx.int32), 0, cache)
    assert output.dtype == mx.bfloat16
    assert all(layer_cache.slice_write_bytes > 0 for layer_cache in cache)


def test_selected_mechanisms_remain_independently_controllable():
    mlx_model = tiny_qwen3_mlx_model()
    disabled = Qwen3ModelWeek2(
        mlx_model,
        checkpoint="selected",
        use_bounded_kv_capacity=False,
        use_register_cached_rms_norm=False,
        use_tiled_prefill_attention=False,
    )
    enabled = Qwen3ModelWeek2(
        mlx_model,
        checkpoint="selected",
        use_bounded_kv_capacity=True,
        use_register_cached_rms_norm=True,
        use_tiled_prefill_attention=True,
    )

    assert not disabled.use_bounded_kv_capacity
    assert not isinstance(disabled.layers_inner[0].input_layernorm, FastRMSNorm)
    assert not disabled.layers_inner[0].self_attn.use_tiled_prefill_attention
    assert enabled.use_bounded_kv_capacity
    assert isinstance(enabled.layers_inner[0].input_layernorm, FastRMSNorm)
    assert enabled.layers_inner[0].self_attn.use_tiled_prefill_attention

    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    actual = enabled(inputs, 0, enabled.create_kv_cache(capacity=3))
    expected = disabled(inputs, 0, disabled.create_kv_cache())
    assert_allclose(actual, expected, mx.bfloat16, atol=3e-2, rtol=3e-2)


def test_retired_experiments_are_not_week2_checkpoints():
    for checkpoint in ("decode-attention", "split-k"):
        try:
            Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint=checkpoint)
        except ValueError as exc:
            assert "unknown Week 2 checkpoint" in str(exc)
        else:
            raise AssertionError(
                f"retired checkpoint {checkpoint!r} remained selectable"
            )


def test_default_model_is_the_cumulative_selected_checkpoint():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model())
    assert model.checkpoint == "selected"
    assert model.mechanism_controls == {
        "capacity_cache": True,
        "register_cached_rmsnorm": True,
        "tiled_prefill": True,
    }
