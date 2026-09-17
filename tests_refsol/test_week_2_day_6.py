"""Week 2 Day 6 long-context decode-attention tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import (
    Qwen3ModelWeek2,
    Qwen3MultiHeadAttention,
    WEEK2_CHECKPOINT_FEATURES,
    long_context_attention,
    scaled_dot_product_attention,
    scaled_dot_product_attention_grouped,
    tiny_llm_ext,
)
from .utils import assert_allclose, tiny_qwen3_mlx_model


HAS_PHASE_2_EXTENSION = hasattr(tiny_llm_ext, "long_context_attention")
model_module = __import__(Qwen3ModelWeek2.__module__, fromlist=["unused"])
should_use_context_selected_attention = (
    model_module.should_use_context_selected_attention
)
CONTEXT_SELECTED_ATTENTION_MIN_CONTEXT = (
    model_module.CONTEXT_SELECTED_ATTENTION_MIN_CONTEXT
)
CONTEXT_SELECTED_ATTENTION_MAX_CONTEXT = (
    model_module.CONTEXT_SELECTED_ATTENTION_MAX_CONTEXT
)


def _qwen_attention_fixture(
    *, query_length: int = 1, context_length: int = 128, dtype=mx.bfloat16
):
    query = mx.zeros((1, 32, query_length, 128), dtype=dtype)
    key = mx.zeros((1, 8, context_length, 128), dtype=dtype)
    return query, key, mx.zeros_like(key)


def test_context_selected_checkpoint_is_independent_and_legacy_name_is_explicit():
    features = WEEK2_CHECKPOINT_FEATURES["context-selected-attention"]
    assert features.context_selected_attention
    assert features.simdgroup_matmul
    assert not features.prefill_fused_gate_up

    with pytest.raises(ValueError, match="replaced by 'context-selected-attention'"):
        Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="long-context-attention")


@pytest.mark.parametrize("query_length", (1, 2))
@pytest.mark.parametrize("context_length", (8192, 8193, 32768))
def test_context_selected_selector_accepts_bounded_decode_shapes(
    query_length, context_length
):
    query, key, value = _qwen_attention_fixture(
        query_length=query_length, context_length=context_length
    )
    assert should_use_context_selected_attention(
        query, key, value, "causal", enabled=True
    )
    assert CONTEXT_SELECTED_ATTENTION_MIN_CONTEXT == 8192
    assert CONTEXT_SELECTED_ATTENTION_MAX_CONTEXT == 32768


def test_context_selected_selector_falls_back_below_threshold_and_when_disabled():
    below_query, below_key, below_value = _qwen_attention_fixture(context_length=8191)
    edge_query, edge_key, edge_value = _qwen_attention_fixture(context_length=8192)
    assert not should_use_context_selected_attention(
        below_query, below_key, below_value, None, enabled=True
    )
    assert not should_use_context_selected_attention(
        edge_query, edge_key, edge_value, None, enabled=False
    )
    assert should_use_context_selected_attention(
        edge_query, edge_key, edge_value, None, enabled=True
    )


def test_context_selected_selector_rejects_mask_dtype_head_layout_and_tail():
    query, key, value = _qwen_attention_fixture(context_length=8192)
    explicit_mask = mx.zeros((1, 1, 1, 8192), dtype=mx.float32)
    assert not should_use_context_selected_attention(
        query, key, value, explicit_mask, enabled=True
    )

    long_query, key, value = _qwen_attention_fixture(
        query_length=3, context_length=8192
    )
    assert not should_use_context_selected_attention(
        long_query, key, value, "causal", enabled=True
    )

    fp32_query, fp32_key, fp32_value = _qwen_attention_fixture(
        context_length=8192, dtype=mx.float32
    )
    assert not should_use_context_selected_attention(
        fp32_query, fp32_key, fp32_value, None, enabled=True
    )
    assert not should_use_context_selected_attention(
        query[:, :16], key, value, None, enabled=True
    )
    assert not should_use_context_selected_attention(
        query.transpose(0, 2, 1, 3), key, value, None, enabled=True
    )
    assert not should_use_context_selected_attention(
        query, key, value[:, :, :-1], None, enabled=True
    )


def test_disable_control_and_counters_are_learner_visible():
    enabled = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(), checkpoint="context-selected-attention"
    )
    disabled = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(),
        checkpoint="context-selected-attention",
        disable_context_selected_attention=True,
    )

    assert enabled.layers_inner[0].self_attn.use_context_selected_attention
    assert not disabled.layers_inner[0].self_attn.use_context_selected_attention
    assert enabled.dispatch_counters() == {
        "context_selected_attention": 0,
        "readable_attention": 0,
        "prefill_fused_gate_up": 0,
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
            self.key = mx.zeros((1, 8, 8192, 128), dtype=mx.bfloat16)
            self.value = mx.zeros_like(self.key)

        def update_and_fetch(self, key, value, *, mask_length, mask):
            assert key.shape == value.shape == (1, 8, 1, 128)
            assert mask_length == 1
            return self.key, self.value, 0, mask

    hidden = mx.zeros((1, 1, 4096), dtype=mx.bfloat16)
    attention(hidden, 8191, Cache())
    explicit_mask = mx.zeros((1, 1, 1, 8192), dtype=mx.float32)
    attention(hidden, 8191, Cache(), explicit_mask)
    attention.use_context_selected_attention = False
    attention(hidden, 8191, Cache())

    assert calls == [
        ("candidate", 8192, False),
        ("readable", 8192, True),
        ("readable", 8192, False),
    ]
    assert offsets == [8191, 8191, 8191, 8191, 8191, 8191]
    assert attention.context_selected_attention_dispatches == 1
    assert attention.readable_attention_dispatches == 2


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
@pytest.mark.parametrize("mask", (None, "causal"))
def test_long_context_extension_matches_readable_selected_decode_gpu(mask):
    query = mx.random.normal((1, 32, 2, 128)).astype(mx.bfloat16)
    key = mx.random.normal((1, 8, 8193, 128)).astype(mx.bfloat16)
    value = mx.random.normal(key.shape).astype(mx.bfloat16)
    result = long_context_attention(query, key, value, 128**-0.5, mask)
    expected = scaled_dot_product_attention_grouped(query, key, value, 128**-0.5, mask)
    assert result.shape == query.shape
    assert result.dtype == mx.bfloat16
    assert_allclose(result, expected, mx.bfloat16, atol=3e-2, rtol=3e-2)
