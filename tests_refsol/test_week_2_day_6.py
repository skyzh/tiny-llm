"""Week 2 shared-input QKV projection reference tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import (
    Qwen3ModelWeek2,
    Qwen3MultiHeadAttention,
    QuantizedWeights,
    WEEK2_CHECKPOINT_FEATURES,
    quantized_qkv,
    supports_shared_input_qkv,
    tiny_llm_ext,
)
from .utils import assert_allclose, tiny_qwen3_mlx_model


HAS_SHARED_INPUT_QKV = hasattr(tiny_llm_ext, "quantized_qkv")


def _quantized_weights(output_dim: int, input_dim: int = 128) -> QuantizedWeights:
    source = mx.random.normal((output_dim, input_dim)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(source, group_size=128, bits=4)
    return QuantizedWeights(scales, biases, 128, 4, packed)


def _projection_oracle(x: mx.array, weight: QuantizedWeights) -> mx.array:
    dequantized = mx.dequantize(
        weight.weight,
        weight.scales,
        weight.biases,
        group_size=weight.group_size,
        bits=weight.bits,
    ).astype(mx.bfloat16)
    return mx.matmul(x.astype(mx.float32), dequantized.astype(mx.float32).T).astype(
        mx.bfloat16
    )


def test_shared_input_checkpoint_order_and_migration_are_explicit():
    assert tuple(WEEK2_CHECKPOINT_FEATURES)[-3:] == (
        "shared-input-qkv",
        "shared-input-gate-up-swiglu",
        "io-aware-dense-attention",
    )
    qkv = WEEK2_CHECKPOINT_FEATURES["shared-input-qkv"]
    gate_up = WEEK2_CHECKPOINT_FEATURES["shared-input-gate-up-swiglu"]
    attention = WEEK2_CHECKPOINT_FEATURES["io-aware-dense-attention"]
    assert qkv.shared_input_qkv and not qkv.shared_input_gate_up_swiglu
    assert gate_up.shared_input_qkv and gate_up.shared_input_gate_up_swiglu
    assert attention.shared_input_qkv and attention.io_aware_dense_attention

    with pytest.raises(ValueError, match="replaced by 'io-aware-dense-attention'"):
        Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="context-selected-attention")
    with pytest.raises(ValueError, match="replaced by 'shared-input-gate-up-swiglu'"):
        Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="prefill-fused-gate-up")


def test_completed_week2_defaults_to_the_final_executable_checkpoint():
    default = Qwen3ModelWeek2(tiny_qwen3_mlx_model())

    assert default.checkpoint == "io-aware-dense-attention"
    assert default.layers_inner[0].self_attn.use_shared_input_qkv
    assert default.layers_inner[0].self_attn.use_io_aware_dense_attention
    assert default.layers_inner[0].mlp.use_shared_input_gate_up_swiglu


@pytest.mark.parametrize("rows", (1, 31, 32, 33, 2048))
def test_shared_input_qkv_selector_accepts_decode_prefill_and_tails(rows):
    q = _quantized_weights(136)
    k = _quantized_weights(64)
    v = _quantized_weights(72)
    assert supports_shared_input_qkv(mx.zeros((rows, 128), dtype=mx.bfloat16), q, k, v)


def test_shared_input_qkv_selector_rejects_dtype_row_and_metadata_boundaries():
    q = _quantized_weights(136)
    k = _quantized_weights(64)
    v = _quantized_weights(72)
    assert not supports_shared_input_qkv(mx.zeros((1, 128), dtype=mx.float32), q, k, v)
    assert not supports_shared_input_qkv(
        mx.zeros((2049, 128), dtype=mx.bfloat16), q, k, v
    )
    wrong_input = _quantized_weights(64, input_dim=256)
    assert not supports_shared_input_qkv(
        mx.zeros((32, 128), dtype=mx.bfloat16), q, wrong_input, v
    )


def test_shared_input_qkv_wrapper_rejects_before_extension_dispatch():
    q = _quantized_weights(136)
    k = _quantized_weights(64)
    v = _quantized_weights(72)
    with pytest.raises(ValueError, match="group-128"):
        quantized_qkv(mx.zeros((1, 128), dtype=mx.float32), q, k, v)


def test_attention_dispatch_selects_shared_or_separate_projection(monkeypatch):
    module = __import__(Qwen3MultiHeadAttention.__module__, fromlist=["unused"])
    calls = []
    projections = (
        mx.zeros((1, 1, 8), dtype=mx.bfloat16),
        mx.zeros((1, 1, 4), dtype=mx.bfloat16),
        mx.zeros((1, 1, 4), dtype=mx.bfloat16),
    )
    monkeypatch.setattr(module, "supports_shared_input_qkv", lambda *_args: True)
    monkeypatch.setattr(
        module,
        "quantized_qkv",
        lambda *_args: calls.append("shared") or projections,
    )

    def project(x, weight):
        calls.append(weight)
        return {"q": projections[0], "k": projections[1], "v": projections[2]}.get(
            weight, x
        )

    monkeypatch.setattr(module, "_linear", project)

    class Cache:
        def update_and_fetch(self, key, value, *, mask_length, mask):
            return key, value, 0, mask

    def attention(enabled):
        layer = Qwen3MultiHeadAttention(
            hidden_size=8,
            num_heads=2,
            num_kv_heads=1,
            head_dim=4,
            wq="q",
            wk="k",
            wv="v",
            wo="o",
            q_norm=mx.ones((4,), dtype=mx.bfloat16),
            k_norm=mx.ones((4,), dtype=mx.bfloat16),
            use_shared_input_qkv=enabled,
            use_io_aware_dense_attention=False,
        )
        layer.q_norm = lambda value: value
        layer.k_norm = lambda value: value
        layer.rope = lambda value, *, offset: value
        return layer

    shared = attention(True)
    separate = attention(False)
    x = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
    shared(x, 0, Cache())
    separate(x, 0, Cache())
    assert shared.shared_input_qkv_dispatches == 1
    assert shared.separate_qkv_dispatches == 0
    assert separate.shared_input_qkv_dispatches == 0
    assert separate.separate_qkv_dispatches == 1
    assert calls == ["shared", "o", "q", "k", "v", "o"]


@pytest.mark.skipif(
    not HAS_SHARED_INPUT_QKV,
    reason="Reference extension build requires the optional Xcode Metal Toolchain",
)
@pytest.mark.parametrize("rows", (1, 33, 64))
def test_shared_input_qkv_matches_three_separate_projections_gpu(rows):
    mx.random.seed(7)
    q = _quantized_weights(136)
    k = _quantized_weights(64)
    v = _quantized_weights(72)
    x = mx.random.normal((rows, 128)).astype(mx.bfloat16)
    actual = quantized_qkv(x, q, k, v)
    expected = tuple(_projection_oracle(x, weight) for weight in (q, k, v))
    assert tuple(item.shape for item in actual) == (
        (rows, 136),
        (rows, 64),
        (rows, 72),
    )
    for result, oracle in zip(actual, expected, strict=True):
        assert result.dtype == mx.bfloat16
        assert_allclose(result, oracle, mx.bfloat16, atol=1e-2, rtol=1e-3)
    # This catches a projection-order mutation even when all three shapes happen to fit.
    assert not mx.array_equal(actual[1], actual[2][:, :64]).item()
