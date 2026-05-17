import mlx.core as mx
import mlx.nn as nn
import tiny_llm_ref
from .utils import assert_allclose
import numpy as np


def get_test_matmul_data():
    # Representative large-model matrix size
    init = nn.init.he_uniform(mx.bfloat16)
    w = init(mx.zeros((512, 3584)))
    x = init(mx.zeros((300, 3584)))
    w_q, scales, biases = mx.quantize(w, group_size=128, bits=4)
    res = mx.quantized_matmul(
        x, w_q, scales=scales, biases=biases, group_size=128, bits=4
    )
    return w_q, scales, biases, x, res


def test_mlx_quantized_matmul(benchmark):
    with mx.stream(mx.gpu):
        w_q, scales, biases, x, res = get_test_matmul_data()
        result = benchmark(
            lambda: mx.quantized_matmul(
                x, w_q, scales=scales, biases=biases, group_size=128, bits=4
            )
        )
        assert_allclose(result, res, precision=mx.bfloat16, rtol=1e-2)


def test_refsol_quantized_matmul(benchmark):
    with mx.stream(mx.gpu):
        w_q, scales, biases, x, res = get_test_matmul_data()
        result = benchmark(
            lambda: tiny_llm_ref.quantized_matmul(
                scales, biases, 128, 4, x, w_q, transpose_b=True
            )
        )
        assert_allclose(result, res, precision=mx.bfloat16, rtol=1e-2)
