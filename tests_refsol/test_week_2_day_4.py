"""Week 2 Day 4 fused-model primitive tests."""

import importlib
import sys
import types

import mlx.core as mx

from .utils import assert_allclose, tiny_qwen3_mlx_model


def _allow_sparse_starter_collection_without_a_generated_extension():
    if __package__ != "tests":
        return
    extension_package = importlib.import_module("extensions.tiny_llm_ext")
    if hasattr(extension_package, "_ext"):
        return
    module_name = "extensions.tiny_llm_ext._ext"
    missing_extension = types.ModuleType(module_name)
    sys.modules[module_name] = missing_extension
    setattr(extension_package, "_ext", missing_extension)


def _load_week_2_symbols():
    _allow_sparse_starter_collection_without_a_generated_extension()
    from .tiny_llm_base import FastRMSNorm, FastRoPE, Qwen3ModelWeek2, swiglu

    return FastRMSNorm, FastRoPE, Qwen3ModelWeek2, swiglu


FastRMSNorm, FastRoPE, Qwen3ModelWeek2, swiglu = _load_week_2_symbols()


def test_task_1_register_cached_rmsnorm_matches_readable_operator():
    x = mx.random.normal((2, 3, 16)).astype(mx.bfloat16)
    weight = mx.random.normal((16,)).astype(mx.bfloat16)
    fast = FastRMSNorm(16, weight, eps=1e-5)
    result = fast(x)
    expected = mx.fast.rms_norm(x, weight, 1e-5)

    assert result is not None, "implement the FastRMSNorm learner seam"
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)
    assert fast.dispatch_counts == {"register_cached": 1, "fixed_width_fallback": 0}


def test_task_2_rope_matches_readable_operator():
    x = mx.random.normal((2, 4, 2, 16)).astype(mx.bfloat16)
    fast_rope = FastRoPE(16, 32, base=10000)
    actual = fast_rope(x, [3, 7])
    expected = mx.fast.rope(
        x.transpose(0, 2, 1, 3),
        16,
        traditional=False,
        base=10000,
        scale=1.0,
        offset=mx.array([3, 7], dtype=mx.int32),
    ).transpose(0, 2, 1, 3)
    assert_allclose(actual, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


def test_task_3_swiglu_matches_readable_operator():
    gate = mx.random.normal((2, 4, 16)).astype(mx.bfloat16)
    up = mx.random.normal((2, 4, 16)).astype(mx.bfloat16)
    assert_allclose(swiglu(gate, up), gate * mx.sigmoid(gate) * up, mx.bfloat16)


def test_task_4_primitive_checkpoints_are_cumulative_and_real():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="swiglu")
    layer = model.layers_inner[0]
    implementation_rope = importlib.import_module(
        f"{FastRMSNorm.__module__.split('.')[0]}.positional_encoding"
    )

    assert model.use_bounded_kv_capacity
    assert layer.self_attn.wq.use_simdgroup_matmul
    assert isinstance(model.norm, FastRMSNorm)
    assert isinstance(layer.input_layernorm, FastRMSNorm)
    assert isinstance(layer.self_attn.rope, FastRoPE)
    assert layer.mlp.use_fast_swiglu

    rms_only = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="rmsnorm")
    rms_layer = rms_only.layers_inner[0]
    assert isinstance(rms_only.norm, FastRMSNorm)
    assert type(rms_layer.self_attn.rope) is implementation_rope.RoPE
    assert not rms_layer.mlp.use_fast_swiglu
