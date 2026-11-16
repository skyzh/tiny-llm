"""Week 2 Day 2 packed-weight tests."""

import mlx.core as mx
import pytest

from benches.bench_course_progression import WEEK2_VARIANTS
from benches.profile_week2_kernels import DEFAULT_CASES
from benches.week2_gpudebug import KNOWN_CHECKPOINTS
from tiny_llm_ref.qwen3_week2 import WEEK2_CHECKPOINTS

from .tiny_llm_base import (
    Qwen3ModelWeek2,
    QuantizedEmbedding,
    QuantizedWeights,
    dequantize_weights,
    quantized_matmul,
)
from .utils import assert_allclose, tiny_qwen3_mlx_model


def _fixed_fixture(seed: int):
    return tiny_qwen3_mlx_model(seed=seed)


def test_task_1_quantized_embedding_dequantizes_selected_rows():
    weight = mx.random.normal((7, 256)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
    embedding = QuantizedEmbedding(
        7, 256, QuantizedWeights(scales, biases, 128, 4, packed)
    )
    indices = mx.array([[1, 4]])

    result = embedding(indices)
    assert result is not None, "implement the QuantizedEmbedding.__call__ learner seam"
    expected = mx.dequantize(
        packed[indices], scales[indices], biases[indices], group_size=128, bits=4
    )
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


def test_task_1_quantized_embedding_accepts_sampled_uint32_tokens():
    weight = mx.random.normal((7, 256)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
    embedding = QuantizedEmbedding(
        7, 256, QuantizedWeights(scales, biases, 128, 4, packed)
    )
    indices = mx.array([[1, 4]], dtype=mx.uint32)

    result = embedding(indices)
    assert result is not None, "implement the QuantizedEmbedding.__call__ learner seam"
    expected = mx.dequantize(
        packed[indices], scales[indices], biases[indices], group_size=128, bits=4
    )
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


def test_task_1_dequantize_weights_matches_packed_weight_values():
    weight = mx.sin(mx.arange(5 * 256, dtype=mx.float32) * 0.07).reshape(5, 256)
    packed, scales, biases = mx.quantize(
        weight.astype(mx.bfloat16), group_size=128, bits=4
    )
    actual = dequantize_weights(packed, scales, biases, 128, 4)
    expected = mx.dequantize(packed, scales, biases, group_size=128, bits=4)
    assert actual is not None, "implement the dequantize_weights learner seam"
    assert actual.shape == (5, 256)
    assert_allclose(actual, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("seed", (0, 2))
def test_task_4_quantized_model_matches_packed_mlx_control(seed: int):
    fixture = _fixed_fixture(seed)
    model = Qwen3ModelWeek2(fixture, checkpoint="packed-weights")
    control = Qwen3ModelWeek2(
        fixture, checkpoint="packed-weights", use_mlx_quantized_linear=True
    )
    tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
    with mx.stream(mx.gpu):
        actual_cache = model.create_kv_cache(capacity=4)
        expected_cache = control.create_kv_cache(capacity=4)
        actual = model(tokens, 0, actual_cache)
        expected = control(tokens, 0, expected_cache)
        decoded = model(mx.array([[4]], dtype=mx.int32), 3, actual_cache)
        decoded_control = control(mx.array([[4]], dtype=mx.int32), 3, expected_cache)
        mx.eval(actual, expected, decoded, decoded_control)

    assert actual.dtype == expected.dtype == mx.bfloat16
    assert actual.shape == expected.shape == (1, 3, fixture.args.vocab_size)
    assert_allclose(actual, expected, mx.bfloat16, atol=1.0, rtol=0.05)
    assert decoded.shape == decoded_control.shape == (1, 1, fixture.args.vocab_size)
    assert_allclose(decoded, decoded_control, mx.bfloat16, atol=1.0, rtol=0.05)


def quantized_matmul_helper(
    stream: mx.Stream,
    precision: mx.Dtype,
    identity_matrix: bool,
):
    with mx.stream(stream):
        group_size = 128
        if identity_matrix:
            input = mx.eye(group_size, dtype=precision)
        else:
            input = mx.random.normal(shape=(3, group_size), dtype=precision)
        weight = mx.random.normal(shape=(5, group_size), dtype=precision)
        w_q, scales, biases = mx.quantize(weight, group_size=group_size, bits=4)
        user_out = quantized_matmul(
            scales=scales,
            biases=biases,
            group_size=group_size,
            bits=4,
            a=input,
            b=w_q,
            transpose_b=True,
        )
        ref_out = mx.quantized_matmul(
            input,
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=4,
            transpose=True,
        )
        assert user_out is not None, "implement the quantized_matmul learner seam"
        assert user_out.dtype == mx.bfloat16
        if identity_matrix:
            assert_allclose(user_out, ref_out, precision)
        else:
            assert_allclose(
                user_out,
                ref_out,
                precision,
                atol=5.0e-1,
                message=f"quantized matmul {precision} comparison",
            )


def test_task_3_quantized_matmul_simple_bf16_gpu():
    quantized_matmul_helper(mx.gpu, mx.bfloat16, True)


def test_task_3_quantized_matmul_complex_bf16_gpu():
    quantized_matmul_helper(mx.gpu, mx.bfloat16, False)


def test_task_4_packed_model_uses_the_simple_kernel_for_prefill_and_decode(monkeypatch):
    import importlib

    quantize = importlib.import_module(QuantizedWeights.__module__)
    original = quantize.quantized_matmul
    observed = []

    def simple(*args, **kwargs):
        observed.append(kwargs.get("use_simdgroup", False))
        return original(*args, **kwargs)

    def premature_matvec(*args, **kwargs):
        pytest.fail("packed-weights must run before the Day 3 matvec is implemented")

    monkeypatch.setattr(quantize, "quantized_matmul", simple)
    monkeypatch.setattr(quantize, "quantized_matvec_custom", premature_matvec)
    fixture = _fixed_fixture(3)
    model = Qwen3ModelWeek2(fixture, checkpoint="packed-weights")
    cache = model.create_kv_cache()
    mx.eval(model(mx.array([[1, 2, 3]], dtype=mx.int32), 0, cache))
    mx.eval(model(mx.array([[4]], dtype=mx.int32), 3, cache))
    assert observed and not any(observed)


def test_day2_public_selectors_include_the_runnable_packed_checkpoint():
    assert WEEK2_CHECKPOINTS[:2] == ("kv-cache", "packed-weights")
    assert KNOWN_CHECKPOINTS[:2] == WEEK2_CHECKPOINTS[:2]
    assert DEFAULT_CASES[:2] == ("kv-cache:decode:128", "packed-weights:decode:128")
    assert [variant.key for variant in WEEK2_VARIANTS][:3] == [
        "week1",
        "week2-kv-cache",
        "week2-packed-weights",
    ]
