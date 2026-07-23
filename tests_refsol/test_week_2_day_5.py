"""Week 2 Day 5 optimized-kernel tests."""

import inspect
import importlib

import mlx.core as mx
import pytest

from tiny_llm_ref.basics import silu
from tiny_llm_ref.layer_norm import RMSNorm
from tiny_llm_ref.positional_encoding import RoPE
from .tiny_llm_base import FastRMSNorm, FastRoPE, Qwen3ModelWeek2, swiglu
from .utils import assert_allclose, tiny_qwen3_mlx_model

implementation_package = FastRMSNorm.__module__.split(".")[0]
week2_kernels = importlib.import_module(FastRMSNorm.__module__)
week1_model = importlib.import_module(f"{implementation_package}.qwen3_week1")
implementation_norm = importlib.import_module(f"{implementation_package}.layer_norm")
implementation_rope = importlib.import_module(
    f"{implementation_package}.positional_encoding"
)


def test_week2_fast_operators_are_course_owned():
    source = inspect.getsource(week2_kernels)
    assert "mx.fast" not in source


def test_fast_rms_norm_matches_week1_implementation():
    x = mx.random.normal((2, 3, 16)).astype(mx.bfloat16)
    weight = mx.random.normal((16,)).astype(mx.bfloat16)
    expected = RMSNorm(16, weight, eps=1e-5)(x)
    result = FastRMSNorm(16, weight, eps=1e-5)(x)
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("offsets", [3, [3, 7]])
def test_fast_rope_matches_week1_implementation(offsets):
    batch_size = 1 if isinstance(offsets, int) else len(offsets)
    seq_len = 4
    x = mx.random.normal((batch_size, seq_len, 2, 16)).astype(mx.float32)
    fast = FastRoPE(16, 32, base=10000)
    readable = RoPE(16, 32, base=10000)
    readable_offsets = (
        slice(offsets, offsets + seq_len)
        if isinstance(offsets, int)
        else [slice(offset, offset + seq_len) for offset in offsets]
    )
    result = fast(x, offsets)
    expected = readable(x, readable_offsets)
    assert_allclose(result, expected, mx.float32, atol=1e-5, rtol=1e-5)


def test_swiglu_matches_readable_expression():
    gate = mx.random.normal((2, 4, 16)).astype(mx.float32)
    up = mx.random.normal((2, 4, 16)).astype(mx.float32)
    assert_allclose(swiglu(gate, up), silu(gate) * up, mx.float32)


def test_completed_model_integrates_all_cumulative_kernels():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="swiglu")
    layer = model.layers_inner[0]

    assert isinstance(model.norm, FastRMSNorm)
    assert isinstance(layer.input_layernorm, FastRMSNorm)
    assert isinstance(layer.self_attn.rope, FastRoPE)
    assert layer.self_attn.use_decode_attention
    assert layer.mlp.use_fast_swiglu


def test_rmsnorm_checkpoint_does_not_enable_later_fast_kernels():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="rmsnorm")
    layer = model.layers_inner[0]

    assert isinstance(model.norm, FastRMSNorm)
    assert isinstance(layer.input_layernorm, FastRMSNorm)
    assert type(layer.self_attn.rope) is implementation_rope.RoPE
    assert layer.self_attn.use_decode_attention
    assert not layer.mlp.use_fast_swiglu


def test_rope_checkpoint_does_not_enable_swiglu_early():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="rope")
    layer = model.layers_inner[0]

    assert isinstance(model.norm, FastRMSNorm)
    assert isinstance(layer.input_layernorm, FastRMSNorm)
    assert isinstance(layer.self_attn.rope, FastRoPE)
    assert layer.self_attn.use_decode_attention
    assert not layer.mlp.use_fast_swiglu


def test_week1_keeps_readable_kernels():
    hidden_size = 16
    num_heads = 2
    num_kv_heads = 1
    head_dim = 8
    attention = week1_model.Qwen3MultiHeadAttention(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        mx.zeros((num_heads * head_dim, hidden_size)),
        mx.zeros((num_kv_heads * head_dim, hidden_size)),
        mx.zeros((num_kv_heads * head_dim, hidden_size)),
        mx.zeros((hidden_size, num_heads * head_dim)),
        mx.ones((head_dim,)),
        mx.ones((head_dim,)),
    )
    assert type(attention.rope) is implementation_rope.RoPE
    assert type(attention.q_norm) is implementation_norm.RMSNorm

    mlp = week1_model.Qwen3MLP(
        hidden_size,
        hidden_size * 2,
        mx.zeros((hidden_size * 2, hidden_size)),
        mx.zeros((hidden_size * 2, hidden_size)),
        mx.zeros((hidden_size, hidden_size * 2)),
    )
    assert mlp(mx.ones((1, 1, hidden_size))).shape == (1, 1, hidden_size)
