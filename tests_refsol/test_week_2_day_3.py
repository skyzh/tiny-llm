"""Week 2 Day 3 SIMD quantized-kernel tests."""

import mlx.core as mx

from .tiny_llm_base import (
    quantized_matmul,
    quantized_matmul_vanilla,
    quantized_matvec_custom,
)
from .utils import assert_allclose


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


def test_task_3_quantized_matmul_simple_f16_gpu():
    quantized_matmul_helper(mx.gpu, mx.float16, True)


def test_task_3_quantized_matmul_complex_f16_gpu():
    quantized_matmul_helper(mx.gpu, mx.float16, False)


def test_task_3_optimized_matvec_matches_vanilla_gpu():
    """The scalar baseline must remain callable for a decode-shaped input."""
    with mx.stream(mx.gpu):
        input = mx.random.normal((1, 256)).astype(mx.bfloat16)
        weight = mx.random.normal((96, 256)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        optimized = quantized_matvec_custom(
            scales, biases, 128, 4, input, packed, transpose_b=True
        )
        vanilla = quantized_matmul_vanilla(
            scales, biases, 128, 4, input, packed, transpose_b=True
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
        assert_allclose(user_out, ref_out, mx.bfloat16, atol=5.0e-1)


def test_task_4_quantized_matvec_custom_m1_gpu():
    quantized_matvec_custom_helper(1)


def test_task_4_quantized_matvec_custom_m8_gpu():
    quantized_matvec_custom_helper(8)
