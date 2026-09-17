"""Week 2 Day 7 fused packed-W4 gate/up + SwiGLU tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import (
    Qwen3MLP,
    Qwen3ModelWeek2,
    QuantizedWeights,
    WEEK2_CHECKPOINT_FEATURES,
    quantized_gate_up_swiglu,
    tiny_llm_ext,
)
from .utils import assert_allclose, tiny_qwen3_mlx_model


HAS_PHASE_2_EXTENSION = hasattr(tiny_llm_ext, "quantized_gate_up_swiglu")
kernel_module = __import__(quantized_gate_up_swiglu.__module__, fromlist=["unused"])
supports_prefill_fused_gate_up = kernel_module.supports_prefill_fused_gate_up


def _quantized_weights(output_dim: int = 136, input_dim: int = 128):
    source = mx.random.normal((output_dim, input_dim)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(source, group_size=128, bits=4)
    return QuantizedWeights(scales, biases, 128, 4, packed), source


def _fused_gate_up_oracle(
    x: mx.array, gate: QuantizedWeights, up: QuantizedWeights
) -> mx.array:
    """Mirror the fused kernel's BF16 weights and FP32 accumulation contract."""
    gate_weight = mx.dequantize(
        gate.weight,
        gate.scales,
        gate.biases,
        group_size=gate.group_size,
        bits=gate.bits,
    ).astype(mx.bfloat16)
    up_weight = mx.dequantize(
        up.weight,
        up.scales,
        up.biases,
        group_size=up.group_size,
        bits=up.bits,
    ).astype(mx.bfloat16)
    x_fp32 = x.astype(mx.float32)
    gate_result = mx.matmul(x_fp32, gate_weight.astype(mx.float32).T)
    up_result = mx.matmul(x_fp32, up_weight.astype(mx.float32).T)
    return (mx.sigmoid(gate_result) * gate_result * up_result).astype(mx.bfloat16)


def test_prefill_fused_checkpoint_is_independent_and_legacy_name_is_explicit():
    features = WEEK2_CHECKPOINT_FEATURES["prefill-fused-gate-up"]
    assert features.prefill_fused_gate_up
    assert features.simdgroup_matmul
    assert not features.context_selected_attention

    with pytest.raises(ValueError, match="replaced by 'prefill-fused-gate-up'"):
        Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="fused-gate-up")


@pytest.mark.parametrize("rows", (32, 128, 512, 2048))
def test_prefill_fused_gate_up_selector_accepts_prefill_rows(rows):
    gate, _ = _quantized_weights()
    up, _ = _quantized_weights()
    x = mx.zeros((rows, 128), dtype=mx.bfloat16)
    assert supports_prefill_fused_gate_up(x, gate, up)


@pytest.mark.parametrize("rows", (1, 31, 2049))
def test_prefill_fused_gate_up_selector_rejects_decode_and_boundary_rows(rows):
    gate, _ = _quantized_weights()
    up, _ = _quantized_weights()
    assert not supports_prefill_fused_gate_up(
        mx.zeros((rows, 128), dtype=mx.bfloat16), gate, up
    )


def test_prefill_fused_gate_up_selector_rejects_unsupported_metadata_and_tail():
    gate, _ = _quantized_weights()
    up, _ = _quantized_weights()
    assert not supports_prefill_fused_gate_up(
        mx.zeros((32, 128), dtype=mx.float32), gate, up
    )

    mismatched, _ = _quantized_weights(output_dim=128)
    assert not supports_prefill_fused_gate_up(
        mx.zeros((32, 128), dtype=mx.bfloat16), gate, mismatched
    )
    assert not supports_prefill_fused_gate_up(
        mx.zeros((32, 127), dtype=mx.bfloat16), gate, up
    )


def test_mlp_dispatches_fused_or_separate_path_with_independent_counters(monkeypatch):
    module = __import__(Qwen3MLP.__module__, fromlist=["unused"])
    calls = []
    monkeypatch.setattr(
        module,
        "supports_prefill_fused_gate_up",
        lambda x, *_args: x.shape[0] >= 32,
    )
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

    fused = Qwen3MLP(
        4,
        4,
        "gate",
        "up",
        "down",
        use_fast_swiglu=False,
        use_prefill_fused_gate_up=True,
    )
    separate = Qwen3MLP(
        4,
        4,
        "gate",
        "up",
        "down",
        use_fast_swiglu=False,
        use_prefill_fused_gate_up=False,
    )
    decode = mx.ones((1, 4), dtype=mx.bfloat16)
    prefill = mx.ones((32, 4), dtype=mx.bfloat16)
    fused(decode)
    fused(prefill)
    separate(prefill)

    assert fused.prefill_fused_gate_up_dispatches == 1
    assert fused.separate_gate_up_dispatches == 1
    assert separate.prefill_fused_gate_up_dispatches == 0
    assert separate.separate_gate_up_dispatches == 1
    assert calls == [
        "gate",
        "up",
        "down",
        "fused",
        "down",
        "gate",
        "up",
        "down",
    ]


def test_disable_control_preserves_separate_gate_up_and_down_projection():
    enabled = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(), checkpoint="prefill-fused-gate-up"
    )
    disabled = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(),
        checkpoint="prefill-fused-gate-up",
        disable_prefill_fused_gate_up=True,
    )
    assert enabled.layers_inner[0].mlp.use_prefill_fused_gate_up
    assert not disabled.layers_inner[0].mlp.use_prefill_fused_gate_up
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
    mx.random.seed(2)
    gate, _ = _quantized_weights()
    up, _ = _quantized_weights()
    x = mx.random.normal((rows, 128)).astype(mx.bfloat16)
    result = quantized_gate_up_swiglu(x, gate, up)
    expected = _fused_gate_up_oracle(x, gate, up)
    assert result.shape == (rows, 136)
    assert result.dtype == mx.bfloat16
    assert_allclose(result, expected, mx.bfloat16, atol=1e-2, rtol=1e-3)
