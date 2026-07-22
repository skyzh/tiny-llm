import inspect

import mlx.core as mx
import pytest

from tiny_llm_ref.basics import silu
from tiny_llm_ref.layer_norm import RMSNorm
from tiny_llm_ref.positional_encoding import RoPE
from tiny_llm_ref.qwen3_week1 import (
    Qwen3MLP as Qwen3MLPWeek1,
    Qwen3MultiHeadAttention as Qwen3MultiHeadAttentionWeek1,
)
from tiny_llm_ref.qwen3_week3 import (
    Qwen3MultiHeadAttention as Qwen3MultiHeadAttentionWeek3,
)
from tiny_llm_ref.quantize import QuantizedWeights
import tiny_llm_ref.week2_kernels as week2_kernels
from tiny_llm_ref.week2_kernels import FastRMSNorm, FastRoPE, swiglu

from .utils import assert_allclose


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


def test_week1_keeps_readable_kernels():
    hidden_size = 16
    num_heads = 2
    num_kv_heads = 1
    head_dim = 8
    attention = Qwen3MultiHeadAttentionWeek1(
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
    assert type(attention.rope) is RoPE
    assert type(attention.q_norm) is RMSNorm

    mlp = Qwen3MLPWeek1(
        hidden_size,
        hidden_size * 2,
        mx.zeros((hidden_size * 2, hidden_size)),
        mx.zeros((hidden_size * 2, hidden_size)),
        mx.zeros((hidden_size, hidden_size * 2)),
    )
    assert mlp(mx.ones((1, 1, hidden_size))).shape == (1, 1, hidden_size)


def test_week3_retains_week2_optimized_kernels():
    hidden_size = 16
    num_heads = 2
    num_kv_heads = 1
    head_dim = 8

    def weights(output_dims, input_dims):
        return QuantizedWeights(
            mx.ones((output_dims, input_dims // 32)),
            mx.zeros((output_dims, input_dims // 32)),
            32,
            4,
            mx.zeros((output_dims, input_dims // 8), dtype=mx.uint32),
        )

    attention = Qwen3MultiHeadAttentionWeek3(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        weights(num_heads * head_dim, hidden_size),
        weights(num_kv_heads * head_dim, hidden_size),
        weights(num_kv_heads * head_dim, hidden_size),
        weights(hidden_size, num_heads * head_dim),
        mx.ones((head_dim,)),
        mx.ones((head_dim,)),
    )
    assert type(attention.rope) is FastRoPE
    assert type(attention.q_norm) is FastRMSNorm
