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
    mx.eval(q, k, v, res)
    mx.synchronize()
    return q, k, v, res


def evaluate_attention(fn):
    result = fn()
    mx.eval(result)
    mx.synchronize()
    return result


def test_mlx_attention(benchmark):
    with mx.stream(mx.gpu):
        q, k, v, res = get_test_attention_data()
        result = benchmark(
            lambda: evaluate_attention(
                lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
            )
        )
        assert_allclose(result, res, precision=mx.float32, rtol=1e-2)


def test_refsol_attention(benchmark):
    with mx.stream(mx.gpu):
        q, k, v, res = get_test_attention_data()
        result = benchmark(
            lambda: evaluate_attention(
                lambda: tiny_llm_ref.scaled_dot_product_attention_grouped(
                    q, k, v, scale=1.0
                )
            )
        )
        assert_allclose(result, res, precision=mx.float32, rtol=1e-2)


def test_refsol_flash_attention(benchmark):
    with mx.stream(mx.gpu):
        q, k, v, res = get_test_attention_data()
        result = benchmark(
            lambda: evaluate_attention(
                lambda: tiny_llm_ref.flash_attention(q, k, v, scale=1.0)
            )
        )
        assert_allclose(result, res, precision=mx.float32, rtol=1e-2)


def test_refsol_flash_attention_causal(benchmark):
    with mx.stream(mx.gpu):
        q, k, v, _ = get_test_attention_data()
        res = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask="causal")
        mx.eval(res)
        mx.synchronize()
        result = benchmark(
            lambda: evaluate_attention(
                lambda: tiny_llm_ref.flash_attention(
                    q, k, v, scale=1.0, mask="causal"
                )
            )
        )
        assert_allclose(result, res, precision=mx.float32, rtol=1e-2)
