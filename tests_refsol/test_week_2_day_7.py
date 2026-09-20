"""Week 2 shared gate/up and IO-aware dense-attention reference tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import (
    Qwen3MLP,
    Qwen3ModelWeek2,
    QuantizedWeights,
    io_aware_dense_attention,
    quantized_gate_up_swiglu,
    scaled_dot_product_attention_grouped,
    should_use_io_aware_dense_attention,
    supports_fused_gate_up,
    tiny_llm_ext,
)
from .utils import assert_allclose, tiny_qwen3_mlx_model
from tiny_llm_ref.quantize import quantized_linear
from tiny_llm_ref.week2_kernels import swiglu


HAS_REFERENCE_EXTENSION = hasattr(tiny_llm_ext, "quantized_gate_up_swiglu")


def _quantized_weights(output_dim: int = 136, input_dim: int = 128):
    source = mx.random.normal((output_dim, input_dim)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(source, group_size=128, bits=4)
    return QuantizedWeights(scales, biases, 128, 4, packed)


def _fused_gate_up_oracle(
    x: mx.array, gate: QuantizedWeights, up: QuantizedWeights
) -> mx.array:
    def unpack(weight):
        return mx.dequantize(
            weight.weight,
            weight.scales,
            weight.biases,
            group_size=weight.group_size,
            bits=weight.bits,
        ).astype(mx.bfloat16)

    x_fp32 = x.astype(mx.float32)
    gate_result = mx.matmul(x_fp32, unpack(gate).astype(mx.float32).T)
    up_result = mx.matmul(x_fp32, unpack(up).astype(mx.float32).T)
    return (mx.sigmoid(gate_result) * gate_result * up_result).astype(mx.bfloat16)


@pytest.mark.parametrize("rows", (1, 31, 32, 33, 2048))
def test_shared_gate_up_selector_accepts_decode_prefill_and_tails(rows):
    gate = _quantized_weights()
    up = _quantized_weights()
    assert supports_fused_gate_up(mx.zeros((rows, 128), dtype=mx.bfloat16), gate, up)


def test_shared_gate_up_selector_rejects_dtype_row_and_metadata_boundaries():
    gate = _quantized_weights()
    up = _quantized_weights()
    assert not supports_fused_gate_up(mx.zeros((1, 128), dtype=mx.float32), gate, up)
    assert not supports_fused_gate_up(
        mx.zeros((2049, 128), dtype=mx.bfloat16), gate, up
    )
    assert not supports_fused_gate_up(
        mx.zeros((32, 128), dtype=mx.bfloat16), gate, _quantized_weights(128)
    )


def test_shared_gate_up_dispatch_and_disable_control(monkeypatch):
    module = __import__(Qwen3MLP.__module__, fromlist=["unused"])
    calls = []
    monkeypatch.setattr(module, "supports_fused_gate_up", lambda *_args: True)
    monkeypatch.setattr(
        module,
        "quantized_gate_up_swiglu",
        lambda x, *_args: calls.append("shared") or x,
    )
    monkeypatch.setattr(module, "_linear", lambda x, weight: calls.append(weight) or x)
    shared = Qwen3MLP(
        4,
        4,
        "gate",
        "up",
        "down",
        use_fast_swiglu=False,
        use_shared_input_gate_up_swiglu=True,
    )
    separate = Qwen3MLP(
        4,
        4,
        "gate",
        "up",
        "down",
        use_fast_swiglu=False,
        use_shared_input_gate_up_swiglu=False,
    )
    x = mx.ones((1, 4), dtype=mx.bfloat16)
    shared(x)
    separate(x)
    assert shared.shared_input_gate_up_swiglu_dispatches == 1
    assert separate.shared_input_gate_up_swiglu_dispatches == 0
    assert calls == ["shared", "down", "gate", "up", "down"]

    enabled = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(), checkpoint="shared-input-gate-up-swiglu"
    )
    disabled = Qwen3ModelWeek2(
        tiny_qwen3_mlx_model(),
        checkpoint="shared-input-gate-up-swiglu",
        disable_shared_input_gate_up_swiglu=True,
    )
    assert enabled.layers_inner[0].mlp.use_shared_input_gate_up_swiglu
    assert not disabled.layers_inner[0].mlp.use_shared_input_gate_up_swiglu


def test_io_aware_selector_covers_dense_masks_and_rejects_boundaries():
    query = mx.zeros((1, 4, 3, 16), dtype=mx.bfloat16)
    key = mx.zeros((1, 2, 7, 16), dtype=mx.bfloat16)
    value = mx.zeros_like(key)
    explicit = mx.zeros((1, 1, 3, 7), dtype=mx.float32)
    assert should_use_io_aware_dense_attention(query, key, value, None, enabled=True)
    assert should_use_io_aware_dense_attention(
        query, key, value, "causal", enabled=True
    )
    assert should_use_io_aware_dense_attention(
        query, key, value, explicit, enabled=True
    )
    assert not should_use_io_aware_dense_attention(
        query, key, value, None, enabled=False
    )
    assert not should_use_io_aware_dense_attention(
        query[:, :3], key, value, None, enabled=True
    )
    assert not should_use_io_aware_dense_attention(
        query.astype(mx.int32), key, value, None, enabled=True
    )
    assert not should_use_io_aware_dense_attention(
        query, key[:, :, :2], value[:, :, :2], None, enabled=True
    )


def test_io_aware_wrapper_rejects_rank_dtype_and_mask_before_dispatch():
    query = mx.zeros((1, 4, 2, 16), dtype=mx.bfloat16)
    key = mx.zeros((1, 2, 4, 16), dtype=mx.bfloat16)
    value = mx.zeros_like(key)
    with pytest.raises(ValueError, match="rank-4"):
        io_aware_dense_attention(query[0], key, value, 16**-0.5)
    with pytest.raises(ValueError, match="matching floating dtypes"):
        io_aware_dense_attention(query, key.astype(mx.float32), value, 16**-0.5)
    with pytest.raises(ValueError, match="unsupported attention mask"):
        io_aware_dense_attention(query, key, value, 16**-0.5, "window")


@pytest.mark.skipif(
    not HAS_REFERENCE_EXTENSION,
    reason="Reference extension build requires the optional Xcode Metal Toolchain",
)
@pytest.mark.parametrize("rows", (33, 64))
def test_shared_gate_up_matches_separate_projection_oracle_gpu(rows):
    mx.random.seed(2)
    gate = _quantized_weights()
    up = _quantized_weights()
    x = mx.random.normal((rows, 128)).astype(mx.bfloat16)
    result = quantized_gate_up_swiglu(x, gate, up)
    expected = _fused_gate_up_oracle(x, gate, up)
    assert result.shape == (rows, 136)
    assert_allclose(result, expected, mx.bfloat16, atol=1e-2, rtol=1e-3)


@pytest.mark.skipif(
    not HAS_REFERENCE_EXTENSION,
    reason="Reference extension build requires the optional Xcode Metal Toolchain",
)
@pytest.mark.parametrize("seed", (2, 7))
@pytest.mark.parametrize("output_dim", (128, 136))
def test_shared_gate_up_decode_matches_separate_bf16_schedule_gpu(seed, output_dim):
    mx.random.seed(seed)
    gate = _quantized_weights(output_dim)
    up = _quantized_weights(output_dim)
    x = mx.random.normal((1, 128)).astype(mx.bfloat16)

    result = quantized_gate_up_swiglu(x, gate, up)
    expected = swiglu(quantized_linear(x, gate), quantized_linear(x, up))

    assert result.shape == (1, output_dim)
    assert mx.array_equal(result, expected).item()


@pytest.mark.skipif(
    not HAS_REFERENCE_EXTENSION,
    reason="Reference extension build requires the optional Xcode Metal Toolchain",
)
@pytest.mark.parametrize("mask_kind", (None, "causal", "explicit"))
def test_io_aware_dense_attention_matches_materialized_oracle_gpu(mask_kind):
    mx.random.seed(11)
    query = mx.random.normal((1, 4, 3, 16)).astype(mx.bfloat16)
    key = mx.random.normal((1, 2, 7, 16)).astype(mx.bfloat16)
    value = mx.random.normal(key.shape).astype(mx.bfloat16)
    mask = mask_kind
    if mask_kind == "explicit":
        mask = mx.zeros((1, 1, 3, 7), dtype=mx.float32)
        mask = mask.at[..., 0].add(-mx.inf)
    result = io_aware_dense_attention(query, key, value, 16**-0.5, mask)
    expected = scaled_dot_product_attention_grouped(
        query.astype(mx.float32),
        key.astype(mx.float32),
        value.astype(mx.float32),
        16**-0.5,
        mask,
    ).astype(mx.bfloat16)
    assert result.shape == query.shape
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)

    if mask_kind == "explicit":
        unmasked = io_aware_dense_attention(query, key, value, 16**-0.5, None)
        # A mask-index mutation changes this deterministic witness.
        assert not mx.array_equal(result, unmasked).item()
