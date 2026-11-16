"""Week 2 Day 3 decode-matvec tests."""

import mlx.core as mx
import pytest

from benches.bench_course_progression import WEEK2_VARIANTS
from benches.bench_week2_operators import SECTIONS
from benches.profile_week2_kernels import DEFAULT_CASES
from benches.week2_gpudebug import KNOWN_CHECKPOINTS
from tiny_llm_ref.qwen3_week2 import WEEK2_CHECKPOINTS

from .tiny_llm_base import (
    Qwen3ModelWeek2,
    quantized_matmul,
    quantized_matmul_vanilla,
    quantized_matvec_custom,
)
from .utils import assert_allclose, tiny_qwen3_mlx_model


def _fixed_fixture(seed: int):
    return tiny_qwen3_mlx_model(seed=seed)


def test_task_1_matvec_kernel_matches_vanilla_gpu():
    """The scalar baseline must remain callable for a decode-shaped input."""
    with mx.stream(mx.gpu):
        input = mx.random.normal((1, 256)).astype(mx.bfloat16)
        weight = mx.random.normal((96, 256)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        optimized = quantized_matmul(
            scales, biases, 128, 4, input, packed, transpose_b=True, use_simdgroup=True
        )
        vanilla = quantized_matmul_vanilla(
            scales, biases, 128, 4, input, packed, transpose_b=True
        )
        assert optimized is not None, (
            "finish the Task 1 matvec kernel and native dispatch through the Day 2 operator"
        )
        assert vanilla is not None, (
            "implement the quantized_matmul_vanilla learner seam"
        )
        assert_allclose(optimized, vanilla, mx.bfloat16, atol=0.5, rtol=2e-2)


def quantized_matvec_custom_helper(num_rows: int):
    with mx.stream(mx.gpu):
        group_size = 128
        input = mx.random.normal(shape=(num_rows, group_size), dtype=mx.bfloat16)
        weight = mx.random.normal(shape=(64, group_size), dtype=mx.bfloat16)
        w_q, scales, biases = mx.quantize(weight, group_size=group_size, bits=4)
        user_out = quantized_matvec_custom(
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
        assert user_out is not None, (
            "implement the quantized_matvec_custom learner seam"
        )
        assert_allclose(user_out, ref_out, mx.bfloat16, atol=5.0e-1)


def test_task_2_quantized_matvec_custom_m1_gpu():
    quantized_matvec_custom_helper(1)


def test_task_2_quantized_matvec_custom_m8_gpu():
    quantized_matvec_custom_helper(8)


def test_task_2_quantized_matvec_custom_qwen_shape_gpu():
    with mx.stream(mx.gpu):
        input = mx.random.normal((1, 2560)).astype(mx.bfloat16)
        weight = mx.random.normal((1024, 2560)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        result = quantized_matvec_custom(
            scales, biases, 128, 4, input, packed, transpose_b=True
        )
        expected = mx.quantized_matmul(
            input,
            packed,
            scales,
            biases,
            group_size=128,
            bits=4,
            transpose=True,
        )
        assert result is not None, "implement the quantized_matvec_custom learner seam"
        assert_allclose(result, expected, mx.bfloat16, atol=1.5)


@pytest.mark.parametrize("rows,outputs", [(1, 5), (8, 65)])
def test_day3_matvec_handles_partial_output_tiles(rows: int, outputs: int):
    with mx.stream(mx.gpu):
        x = mx.random.normal((rows, 256)).astype(mx.bfloat16)
        weight = mx.random.normal((outputs, 256)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        actual = quantized_matvec_custom(
            scales, biases, 128, 4, x, packed, transpose_b=True
        )
        assert actual is not None, "implement the quantized_matvec_custom learner seam"
        expected = mx.quantized_matmul(
            x, packed, scales, biases, group_size=128, bits=4, transpose=True
        )
        assert actual.shape == (rows, outputs)
        assert_allclose(actual, expected, mx.bfloat16, atol=5e-1)


def test_day3_public_selectors_preserve_day2_prefix():
    assert WEEK2_CHECKPOINTS[:3] == ("kv-cache", "packed-weights", "quantized-matvec")
    assert KNOWN_CHECKPOINTS[:3] == WEEK2_CHECKPOINTS[:3]
    assert DEFAULT_CASES[:3] == (
        "kv-cache:decode:128",
        "packed-weights:decode:128",
        "quantized-matvec:decode:128",
    )
    assert [variant.key for variant in WEEK2_VARIANTS][:4] == [
        "week1",
        "week2-kv-cache",
        "week2-packed-weights",
        "week2-quantized-matvec",
    ]
    assert SECTIONS[:3] == ("embedding", "decode-projections", "prefill-projections")


@pytest.mark.parametrize("seed", (0, 2))
def test_task_3_quantized_model_matches_packed_mlx_control(seed: int):
    fixture = _fixed_fixture(seed)
    model = Qwen3ModelWeek2(fixture, checkpoint="quantized-matvec")
    control = Qwen3ModelWeek2(
        fixture, checkpoint="quantized-matvec", use_mlx_quantized_linear=True
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
