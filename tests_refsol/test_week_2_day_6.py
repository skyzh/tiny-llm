"""Week 2 Day 6 long-context decode-attention tests."""

import mlx.core as mx
import pytest

from extensions_ref import tiny_llm_ext_ref
from tiny_llm_ref.attention import scaled_dot_product_attention_grouped

from tiny_llm_ref.qwen3_week2 import (
    LONG_CONTEXT_ATTENTION_MAX_CONTEXT,
    Qwen3ModelWeek2,
    Qwen3MultiHeadAttention,
    WEEK2_CHECKPOINT_FEATURES,
    should_use_long_context_attention,
)
from tiny_llm_ref.week2_kernels import (
    long_context_attention,
    scaled_dot_product_attention,
)
from .utils import assert_allclose, tiny_qwen3_mlx_model


HAS_PHASE_2_EXTENSION = hasattr(tiny_llm_ext_ref, "long_context_attention")


def _qwen_attention_fixture(
    *, query_length: int = 1, context_length: int = 128, dtype=mx.bfloat16
):
    query = mx.zeros((1, 32, query_length, 128), dtype=dtype)
    key = mx.zeros((1, 8, context_length, 128), dtype=dtype)
    return query, key, mx.zeros_like(key)


def test_long_context_checkpoint_is_independent_and_legacy_name_is_explicit():
    features = WEEK2_CHECKPOINT_FEATURES["long-context-attention"]
    assert features.long_context_attention
    assert features.simdgroup_matmul
    assert not features.fused_gate_up

    with pytest.raises(ValueError, match="replaced by 'long-context-attention'"):
        Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="decode-attention")


@pytest.mark.parametrize("query_length", (1, 2))
@pytest.mark.parametrize("context_length", (1, 128, 512, 2048, 8192))
def test_long_context_selector_accepts_qwen_decode_shapes(query_length, context_length):
    query, key, value = _qwen_attention_fixture(
        query_length=query_length, context_length=context_length
    )
    assert should_use_long_context_attention(query, key, value, "causal", enabled=True)
    assert LONG_CONTEXT_ATTENTION_MAX_CONTEXT == 32768


def test_long_context_selector_falls_back_for_nearest_unsupported_cases():
    query, key, value = _qwen_attention_fixture()
    explicit_mask = mx.zeros((1, 1, 1, 128), dtype=mx.float32)
    assert not should_use_long_context_attention(query, key, value, None, enabled=False)
    assert not should_use_long_context_attention(
        query, key, value, explicit_mask, enabled=True
    )

    long_query, key, value = _qwen_attention_fixture(query_length=3)
    assert not should_use_long_context_attention(
        long_query, key, value, "causal", enabled=True
    )

    fp32_query, fp32_key, fp32_value = _qwen_attention_fixture(dtype=mx.float32)
    assert not should_use_long_context_attention(
        fp32_query, fp32_key, fp32_value, None, enabled=True
    )


def test_disable_control_and_counters_are_learner_visible():
    enabled = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(), checkpoint="long-context-attention"
    )
    disabled = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(),
        checkpoint="long-context-attention",
        disable_long_context_attention=True,
    )

    assert enabled.layers_inner[0].self_attn.use_long_context_attention
    assert not disabled.layers_inner[0].self_attn.use_long_context_attention
    assert enabled.dispatch_counters() == {
        "long_context_attention": 0,
        "readable_attention": 0,
        "fused_gate_up": 0,
        "separate_gate_up": 0,
    }


def test_attention_dispatch_records_candidate_fallback_and_offset(monkeypatch):
    module = __import__(Qwen3MultiHeadAttention.__module__, fromlist=["unused"])
    calls = []
    projections = {
        "q": mx.zeros((1, 1, 32 * 128), dtype=mx.bfloat16),
        "k": mx.zeros((1, 1, 8 * 128), dtype=mx.bfloat16),
        "v": mx.zeros((1, 1, 8 * 128), dtype=mx.bfloat16),
    }

    def project(x, weight):
        return projections.get(weight, x)

    def candidate(query, key, value, *, scale, mask):
        calls.append(("candidate", key.shape[-2], isinstance(mask, mx.array)))
        return query

    def readable(query, key, value, *, scale, mask):
        calls.append(("readable", key.shape[-2], isinstance(mask, mx.array)))
        return query

    monkeypatch.setattr(module, "_linear", project)
    monkeypatch.setattr(module, "long_context_attention", candidate)
    monkeypatch.setattr(module, "scaled_dot_product_attention_grouped", readable)

    attention = Qwen3MultiHeadAttention(
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        wq="q",
        wk="k",
        wv="v",
        wo="o",
        q_norm=mx.ones((128,), dtype=mx.bfloat16),
        k_norm=mx.ones((128,), dtype=mx.bfloat16),
    )
    offsets = []
    attention.q_norm = lambda value: value
    attention.k_norm = lambda value: value
    attention.rope = lambda value, *, offset: offsets.append(offset) or value

    class Cache:
        def __init__(self):
            self.key = mx.zeros((1, 8, 513, 128), dtype=mx.bfloat16)
            self.value = mx.zeros_like(self.key)

        def update_and_fetch(self, key, value, *, mask_length, mask):
            assert key.shape == value.shape == (1, 8, 1, 128)
            assert mask_length == 1
            return self.key, self.value, 0, mask

    hidden = mx.zeros((1, 1, 4096), dtype=mx.bfloat16)
    attention(hidden, 512, Cache())
    explicit_mask = mx.zeros((1, 1, 1, 513), dtype=mx.float32)
    attention(hidden, 512, Cache(), explicit_mask)

    assert calls == [
        ("candidate", 513, False),
        ("readable", 513, True),
    ]
    assert offsets == [512, 512, 512, 512]
    assert attention.long_context_attention_dispatches == 1
    assert attention.readable_attention_dispatches == 1


def test_readable_fallback_handles_explicit_masks_and_finite_extremes():
    query = mx.array([[[[80.0, -80.0], [40.0, -40.0]]]], dtype=mx.float32)
    key = mx.array([[[[1.0, -1.0], [-1.0, 1.0], [0.5, -0.5]]]], dtype=mx.float32)
    value = mx.array([[[[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]]], dtype=mx.float32)
    mask = mx.array([[[[0.0, -mx.inf, 0.0], [0.0, 0.0, -mx.inf]]]])

    result = scaled_dot_product_attention(query, key, value, 2**-0.5, mask)
    expected = scaled_dot_product_attention_grouped(query, key, value, 2**-0.5, mask)
    mx.eval(result)
    assert mx.all(mx.isfinite(result)).item()
    assert_allclose(result, expected, mx.float32, atol=1e-5, rtol=1e-5)


def test_long_context_wrapper_rejects_masks_and_arbitrary_shapes_before_dispatch():
    query, key, value = _qwen_attention_fixture()
    with pytest.raises(ValueError, match="does not accept explicit masks"):
        long_context_attention(
            query,
            key,
            value,
            128**-0.5,
            mx.zeros((1, 1, 1, 128), dtype=mx.float32),
        )
    with pytest.raises(ValueError, match="Qwen3-4B dense GQA"):
        long_context_attention(
            mx.zeros((1, 4, 1, 128), dtype=mx.bfloat16),
            mx.zeros((1, 1, 128, 128), dtype=mx.bfloat16),
            mx.zeros((1, 1, 128, 128), dtype=mx.bfloat16),
            128**-0.5,
        )


@pytest.mark.skipif(
    not HAS_PHASE_2_EXTENSION,
    reason="Phase 2 extension build requires the optional Xcode Metal Toolchain",
)
def test_long_context_extension_matches_readable_qwen_decode_gpu():
    query = mx.random.normal((1, 32, 2, 128)).astype(mx.bfloat16)
    key = mx.random.normal((1, 8, 129, 128)).astype(mx.bfloat16)
    value = mx.random.normal(key.shape).astype(mx.bfloat16)
    result = long_context_attention(query, key, value, 128**-0.5, "causal")
    expected = scaled_dot_product_attention_grouped(
        query, key, value, 128**-0.5, "causal"
    )
    assert result.shape == query.shape
    assert result.dtype == mx.bfloat16
    assert_allclose(result, expected, mx.bfloat16, atol=3e-2, rtol=3e-2)
