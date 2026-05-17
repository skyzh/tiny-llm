import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


def quantized_matmul_outputs(
    stream: mx.Stream, identity_matrix: bool, group_size: int = 64
):
    with mx.stream(stream):
        mx.random.seed(0)
        input_size = group_size * 4
        if identity_matrix:
            input = mx.eye(input_size, dtype=mx.bfloat16)
        else:
            input = mx.random.normal(shape=(3, input_size), dtype=mx.bfloat16)
        weight = mx.random.normal(shape=(8, input_size), dtype=mx.bfloat16)
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
        dequantized_weight = mx.dequantize(
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=4,
        )
        ref_out = input @ dequantized_weight.T
        return user_out, ref_out


def assert_quantized_matmul_close(user_out: mx.array, ref_out: mx.array):
    assert_allclose(
        user_out,
        ref_out,
        mx.bfloat16,
        atol=2.0e-1,
        message="quantized matmul bf16 comparison",
    )


def test_task_2_quantized_matmul_simple_bf16_cpu():
    user_out, ref_out = quantized_matmul_outputs(mx.cpu, True)
    assert_allclose(user_out, ref_out, mx.bfloat16)


def test_task_2_quantized_matmul_complex_bf16_cpu():
    user_out, ref_out = quantized_matmul_outputs(mx.cpu, False)
    assert_quantized_matmul_close(user_out, ref_out)


def test_task_3_quantized_matmul_simple_bf16_gpu():
    user_out, ref_out = quantized_matmul_outputs(mx.gpu, True)
    assert_allclose(user_out, ref_out, mx.bfloat16)


def test_task_3_quantized_matmul_complex_bf16_gpu():
    user_out, ref_out = quantized_matmul_outputs(mx.gpu, False)
    assert_quantized_matmul_close(user_out, ref_out)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_4_quantized_matmul_qwen3_group_size_128_bf16(stream):
    user_out, ref_out = quantized_matmul_outputs(
        stream, False, group_size=128
    )
    assert_quantized_matmul_close(user_out, ref_out)
