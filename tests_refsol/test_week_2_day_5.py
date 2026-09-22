"""Week 2 Day 5 fused-model primitive tests."""

import importlib

import mlx.core as mx

from tiny_llm_ref.basics import silu
from tiny_llm_ref.layer_norm import RMSNorm
from tiny_llm_ref.positional_encoding import RoPE
from .tiny_llm_base import FastRMSNorm, FastRoPE, Qwen3ModelWeek2, swiglu
from .utils import assert_allclose, tiny_qwen3_mlx_model


def test_task_1_register_cached_rmsnorm_matches_readable_operator():
    x = mx.random.normal((2, 3, 16)).astype(mx.bfloat16)
    weight = mx.random.normal((16,)).astype(mx.bfloat16)
    fast = FastRMSNorm(16, weight, eps=1e-5)
    result = fast(x)
    expected = RMSNorm(16, weight, eps=1e-5)(x)

    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)
    assert fast.dispatch_counts == {"register_cached": 1, "fixed_width_fallback": 0}


def test_task_2_rope_and_swiglu_match_readable_operators():
    x = mx.random.normal((2, 4, 2, 16)).astype(mx.bfloat16)
    fast_rope = FastRoPE(16, 32, base=10000)
    readable_rope = RoPE(16, 32, base=10000)
    actual = fast_rope(x, [3, 7])
    expected = readable_rope(x, [slice(3, 7), slice(7, 11)])
    assert_allclose(actual, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)

    gate = mx.random.normal((2, 4, 16)).astype(mx.bfloat16)
    up = mx.random.normal((2, 4, 16)).astype(mx.bfloat16)
    assert_allclose(swiglu(gate, up), silu(gate) * up, mx.bfloat16)


def test_task_3_primitive_checkpoints_are_cumulative_and_real():
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
