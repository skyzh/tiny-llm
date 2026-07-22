"""Week 2 Day 2 vanilla quantization tests."""

import inspect
import importlib

import mlx.core as mx

from .tiny_llm_base import QuantizedEmbedding, QuantizedWeights, quantized_matmul
from .utils import assert_allclose

embedding_module = importlib.import_module(QuantizedEmbedding.__module__)
quantize_module = importlib.import_module(quantized_matmul.__module__)


def test_task_1_quantized_embedding_dequantizes_selected_rows():
    weight = mx.random.normal((7, 256)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
    embedding = QuantizedEmbedding(
        7, 256, QuantizedWeights(scales, biases, 128, 4, packed)
    )
    indices = mx.array([[1, 4]])

    result = embedding(indices)
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
    expected = mx.dequantize(
        packed[indices], scales[indices], biases[indices], group_size=128, bits=4
    )
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


def test_week2_quantization_path_uses_course_owned_operators():
    source = (
        inspect.getsource(quantize_module.quantized_matmul)
        + inspect.getsource(quantize_module.dequantize_weights)
        + inspect.getsource(embedding_module.QuantizedEmbedding.__call__)
    )
    assert "mx.quantized_matmul" not in source
    assert "mx.dequantize" not in source


def quantized_matmul_cpu_helper(precision: mx.Dtype, identity_matrix: bool):
    with mx.stream(mx.cpu):
        if identity_matrix:
            input = mx.eye(128, dtype=precision)
        else:
            input = mx.random.normal((3, 128), dtype=precision)
        weight = mx.random.normal((5, 128), dtype=precision)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        result = quantized_matmul(
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
        assert_allclose(result, expected, precision, atol=5e-1)


def test_task_2_vanilla_bf16_identity_cpu():
    quantized_matmul_cpu_helper(mx.bfloat16, True)


def test_task_2_vanilla_bf16_random_cpu():
    quantized_matmul_cpu_helper(mx.bfloat16, False)


def test_task_2_vanilla_f16_identity_cpu():
    quantized_matmul_cpu_helper(mx.float16, True)


def test_task_2_vanilla_f16_random_cpu():
    quantized_matmul_cpu_helper(mx.float16, False)
