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


def get_prefill_attention_data(sequence_length):
    # A single Qwen2 7B request with grouped-query attention.
    init = nn.init.he_uniform(mx.float32)
    q = init(mx.zeros((1, 28, sequence_length, 128)))
    k = init(mx.zeros((1, 4, sequence_length, 128)))
    v = init(mx.zeros((1, 4, sequence_length, 128)))
    mx.eval(q, k, v)
    mx.synchronize()
    return q, k, v


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
                lambda: tiny_llm_ref.flash_attention(q, k, v, scale=1.0, mask="causal")
            )
        )
        assert_allclose(result, res, precision=mx.float32, rtol=1e-2)


@pytest.mark.parametrize("sequence_length", [128, 512, 1024, 2048])
def test_mlx_attention_causal_prefill(benchmark, sequence_length):
    with mx.stream(mx.gpu):
        q, k, v = get_prefill_attention_data(sequence_length)
        benchmark(
            lambda: evaluate_attention(
                lambda: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=1.0, mask="causal"
                )
            )
        )


@pytest.mark.parametrize("sequence_length", [128, 512, 1024, 2048])
def test_refsol_attention_causal_prefill(benchmark, sequence_length):
    with mx.stream(mx.gpu):
        q, k, v = get_prefill_attention_data(sequence_length)
        benchmark(
            lambda: evaluate_attention(
                lambda: tiny_llm_ref.scaled_dot_product_attention_grouped(
                    q, k, v, scale=1.0, mask="causal"
                )
            )
        )


@pytest.mark.parametrize("sequence_length", [128, 512, 1024, 2048])
def test_refsol_flash_attention_causal_prefill(benchmark, sequence_length):
    with mx.stream(mx.gpu):
        q, k, v = get_prefill_attention_data(sequence_length)
        benchmark(
            lambda: evaluate_attention(
                lambda: tiny_llm_ref.flash_attention(q, k, v, scale=1.0, mask="causal")
            )
        )
