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


def get_qwen3_4b_prefill_data(sequence_length):
    # Qwen3-4B: 32 query heads, 8 KV heads, and head_dim=128.
    init = nn.init.he_uniform(mx.bfloat16)
    q = init(mx.zeros((1, 32, sequence_length, 128)))
    k = init(mx.zeros((1, 8, sequence_length, 128)))
    v = init(mx.zeros((1, 8, sequence_length, 128)))
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


@pytest.mark.parametrize("sequence_length", [128, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("implementation", ["mlx", "explicit", "simdgroup_matrix"])
def test_qwen3_4b_causal_prefill(benchmark, sequence_length, implementation):
    with mx.stream(mx.gpu):
        q, k, v = get_qwen3_4b_prefill_data(sequence_length)
        if implementation == "mlx":
            fn = lambda: mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1.0, mask="causal"
            )
        elif implementation == "explicit":
            fn = lambda: tiny_llm_ref.scaled_dot_product_attention_grouped(
                q, k, v, scale=1.0, mask="causal"
            )
        else:
            fn = lambda: tiny_llm_ref.flash_attention(q, k, v, scale=1.0, mask="causal")
        benchmark(lambda: evaluate_attention(fn))
