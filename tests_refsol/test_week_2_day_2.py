import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


def quantized_matmul_helper(
    stream: mx.Stream,
    identity_matrix: bool,
    precision: mx.Dtype,
    group_size: int = 64,
    rtol: float | None = None,
    atol: float | None = None,
):
    with mx.stream(stream):
        input_size = group_size * 4
        if identity_matrix:
            input = mx.eye(input_size, dtype=precision)
        else:
            input = mx.random.normal(shape=(3, input_size), dtype=precision)
        weight = mx.random.normal(shape=(8, input_size), dtype=precision)
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
        assert_allclose(user_out, ref_out, precision, rtol=rtol, atol=atol)


def test_task_2_quantized_matmul_simple_f16_cpu():
    quantized_matmul_helper(mx.cpu, True, mx.float16)


def test_task_2_quantized_matmul_complex_f16_cpu():
    quantized_matmul_helper(mx.cpu, False, mx.float16)


def test_task_2_quantized_matmul_simple_f32_cpu():
    quantized_matmul_helper(mx.cpu, True, mx.float32)


def test_task_2_quantized_matmul_complex_f32_cpu():
    quantized_matmul_helper(mx.cpu, False, mx.float32)


def test_task_3_quantized_matmul_simple_f16_gpu():
    quantized_matmul_helper(mx.gpu, True, mx.float16)


def test_task_3_quantized_matmul_complex_f16_gpu():
    quantized_matmul_helper(mx.gpu, False, mx.float16, atol=5e-2)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_4_quantized_matmul_qwen3_checkpoint_group_size_f16(stream):
    quantized_matmul_helper(stream, False, mx.float16, group_size=128, atol=5e-2)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_4_quantized_matmul_qwen3_checkpoint_group_size_bf16(stream):
    quantized_matmul_helper(
        stream, False, mx.bfloat16, group_size=128, rtol=1e-1, atol=5e-1
    )
