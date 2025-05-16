import pytest
import mlx.core as mx
from .tiny_llm_base import *
import numpy as np
from .utils import *


def quantized_matmul_helper(
    stream: mx.Stream, identity_matrix: bool, precision: np.dtype
):
    with mx.stream(stream):
        if identity_matrix:
            input = mx.array(np.eye(64).astype(precision))
        else:
            input = mx.array(np.random.randn(3, 64).astype(precision))
        weight = mx.array(np.random.randn(5, 64).astype(precision))
        w_q, scales, biases = mx.quantize(weight)
        user_out = quantized_matmul(
            scales=scales,
            biases=biases,
            group_size=64,
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
            group_size=64,
            bits=4,
            transpose=True,
        )
        assert_allclose(user_out, ref_out, precision)


def test_task_1_quantized_matmul_simple_f16_cpu():
    quantized_matmul_helper(mx.cpu, True, np.float16)


def test_task_1_quantized_matmul_complex_f16_cpu():
    quantized_matmul_helper(mx.cpu, False, np.float16)


def test_task_2_quantized_matmul_simple_f16_gpu():
    quantized_matmul_helper(mx.gpu, True, np.float16)


def test_task_2_quantized_matmul_complex_f16_gpu():
    quantized_matmul_helper(mx.gpu, False, np.float16)
