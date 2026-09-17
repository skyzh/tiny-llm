"""Week 2 Day 7 fused packed-W4 gate/up + SwiGLU tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import (
    Qwen3MLP,
    Qwen3ModelWeek2,
    QuantizedWeights,
    WEEK2_CHECKPOINT_FEATURES,
    quantized_gate_up_swiglu,
    supports_fused_gate_up,
    tiny_llm_ext,
)
from .utils import assert_allclose, tiny_qwen3_mlx_model


HAS_PHASE_2_EXTENSION = hasattr(tiny_llm_ext, "quantized_gate_up_swiglu")


def _quantized_weights(output_dim: int = 136, input_dim: int = 128):
    source = mx.random.normal((output_dim, input_dim)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(source, group_size=128, bits=4)
    return QuantizedWeights(scales, biases, 128, 4, packed), source


def test_fused_gate_up_checkpoint_is_independent_and_legacy_name_is_explicit():
    features = WEEK2_CHECKPOINT_FEATURES["fused-gate-up"]
    assert features.fused_gate_up
    assert features.simdgroup_matmul
    assert not features.long_context_attention

    with pytest.raises(ValueError, match="replaced by 'fused-gate-up'"):
        Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="split-k")


@pytest.mark.parametrize("rows", (1, 32, 128, 512, 2048))
def test_fused_gate_up_selector_accepts_product_rows_and_tail_output(rows):
    gate, _ = _quantized_weights()
    up, _ = _quantized_weights()
    x = mx.zeros((rows, 128), dtype=mx.bfloat16)
    assert supports_fused_gate_up(x, gate, up)


def test_fused_gate_up_selector_rejects_nearest_unsupported_metadata():
    gate, _ = _quantized_weights()
    up, _ = _quantized_weights()
    assert not supports_fused_gate_up(
        mx.zeros((2049, 128), dtype=mx.bfloat16), gate, up
    )
    assert not supports_fused_gate_up(mx.zeros((32, 128), dtype=mx.float32), gate, up)

    mismatched, _ = _quantized_weights(output_dim=128)
    assert not supports_fused_gate_up(
        mx.zeros((32, 128), dtype=mx.bfloat16), gate, mismatched
    )


def test_mlp_dispatches_fused_or_separate_path_with_independent_counters(monkeypatch):
    module = __import__(Qwen3MLP.__module__, fromlist=["unused"])
    calls = []
    monkeypatch.setattr(module, "supports_fused_gate_up", lambda *_args: True)
    monkeypatch.setattr(
        module,
        "quantized_gate_up_swiglu",
        lambda x, *_args: calls.append("fused") or x,
    )
    monkeypatch.setattr(
        module,
        "_linear",
        lambda x, weight: calls.append(weight) or x,
    )

    fused = Qwen3MLP(4, 4, "gate", "up", "down", use_fused_gate_up=True)
    separate = Qwen3MLP(
        4,
        4,
        "gate",
        "up",
        "down",
        use_fast_swiglu=False,
        use_fused_gate_up=False,
    )
    x = mx.ones((1, 1, 4), dtype=mx.bfloat16)
    fused(x)
    separate(x)

    assert fused.fused_gate_up_dispatches == 1
    assert fused.separate_gate_up_dispatches == 0
    assert separate.fused_gate_up_dispatches == 0
    assert separate.separate_gate_up_dispatches == 1
    assert calls == ["fused", "down", "gate", "up", "down"]


def test_disable_control_preserves_separate_gate_up_and_down_projection():
    enabled = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="fused-gate-up")
    disabled = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(),
        checkpoint="fused-gate-up",
        disable_fused_gate_up=True,
    )
    assert enabled.layers_inner[0].mlp.use_fused_gate_up
    assert not disabled.layers_inner[0].mlp.use_fused_gate_up
    assert enabled.layers_inner[0].mlp.w_down is not None
    assert disabled.layers_inner[0].mlp.w_down is not None


def test_fused_wrapper_rejects_unsupported_metadata_before_dispatch():
    gate, _ = _quantized_weights()
    up, _ = _quantized_weights()
    with pytest.raises(ValueError, match="group-128"):
        quantized_gate_up_swiglu(mx.zeros((1, 128), dtype=mx.float32), gate, up)


@pytest.mark.skipif(
    not HAS_PHASE_2_EXTENSION,
    reason="Phase 2 extension build requires the optional Xcode Metal Toolchain",
)
@pytest.mark.parametrize("rows", (1, 32, 128, 512, 2048))
def test_fused_gate_up_matches_separate_quantized_projections_gpu(rows):
    gate, _ = _quantized_weights()
    up, _ = _quantized_weights()
    x = mx.random.normal((rows, 128)).astype(mx.bfloat16)
    result = quantized_gate_up_swiglu(x, gate, up)
    gate_result = mx.quantized_matmul(
        x,
        gate.weight,
        gate.scales,
        gate.biases,
        transpose=True,
        group_size=128,
        bits=4,
    )
    up_result = mx.quantized_matmul(
        x,
        up.weight,
        up.scales,
        up.biases,
        transpose=True,
        group_size=128,
        bits=4,
    )
    expected = (mx.sigmoid(gate_result) * gate_result * up_result).astype(mx.bfloat16)
    assert result.shape == (rows, 136)
    assert result.dtype == mx.bfloat16
    assert_allclose(result, expected, mx.bfloat16, atol=1.5, rtol=2e-2)
