import pytest
import mlx.core as mx
import numpy as np
from .tiny_llm_base import *
from .utils import *


def round_to_bfloat16(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32).copy()
    bits = (
        bits
        + np.uint32(0x7FFF)
        + ((bits >> np.uint32(16)) & np.uint32(1))
    ) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def quantized_matmul_reference(
    inputs: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
) -> mx.array:
    inputs = round_to_bfloat16(np.array(inputs.astype(mx.float32)))
    weight = np.array(weight)
    scales = round_to_bfloat16(np.array(scales.astype(mx.float32)))
    biases = round_to_bfloat16(np.array(biases.astype(mx.float32)))

    shifts = np.arange(0, 32, bits, dtype=np.uint32)
    mask = np.uint32((1 << bits) - 1)
    unpacked = ((weight[..., None] >> shifts) & mask).astype(np.float32)
    unpacked = unpacked.reshape(weight.shape[0], -1)

    groups_per_row = inputs.shape[1] // group_size
    weight = unpacked.reshape(weight.shape[0], groups_per_row, group_size)
    weight = round_to_bfloat16(
        weight * scales[..., None] + biases[..., None]
    ).reshape(weight.shape[0], -1)

    output = round_to_bfloat16(
        np.zeros((inputs.shape[0], weight.shape[0]), dtype=np.float32)
    )
    for col in range(inputs.shape[1]):
        product = round_to_bfloat16(
            inputs[:, col : col + 1] * weight[:, col][None, :]
        )
        output = round_to_bfloat16(output + product)
    return mx.array(output, dtype=mx.bfloat16)


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
        ref_out = quantized_matmul_reference(
            input,
            w_q,
            scales,
            biases,
            group_size,
            4,
        )
        return user_out, ref_out


def assert_quantized_matmul_close(user_out: mx.array, ref_out: mx.array):
    assert_allclose(user_out, ref_out, mx.bfloat16)


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
