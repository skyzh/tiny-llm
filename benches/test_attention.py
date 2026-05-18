import mlx.core as mx
import mlx.nn as nn
import tiny_llm_ref
from .utils import assert_allclose
import pytest


def get_test_attention_data():
    # Representative large-model attention size
    init = nn.init.he_uniform(mx.float32)
    q = init(mx.zeros((10, 28, 1024, 128)))
    k = init(mx.zeros((10, 4, 1024, 128)))
    v = init(mx.zeros((10, 4, 1024, 128)))
    res = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
    return q, k, v, res


def test_mlx_attention(benchmark):
    with mx.stream(mx.gpu):
        q, k, v, res = get_test_attention_data()
        result = benchmark(
            lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
        )
        assert_allclose(result, res, precision=mx.float32, rtol=1e-2)


def test_refsol_attention(benchmark):
    with mx.stream(mx.gpu):
        q, k, v, res = get_test_attention_data()
        result = benchmark(
            lambda: tiny_llm_ref.scaled_dot_product_attention_grouped(
                q, k, v, scale=1.0
            )
        )
        assert_allclose(result, res, precision=mx.float32, rtol=1e-2)


def test_refsol_flash_attention(benchmark):
    with mx.stream(mx.gpu):
        q, k, v, res = get_test_attention_data()
        result = benchmark(lambda: tiny_llm_ref.flash_attention(q, k, v, scale=1.0))
        assert_allclose(result, res, precision=mx.float32, rtol=1e-2)
