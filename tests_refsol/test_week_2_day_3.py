import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


def attention_helper(stream: mx.Stream, H_q, H, L, E, S, BATCH):
    precision = mx.float32
    with mx.stream(stream):
        q_shape = (BATCH, H_q, L, E)
        kv_shape = (BATCH, H, S, E)
        scale = 0.9
        for _ in range(100):
            query = mx.random.uniform(shape=q_shape, dtype=precision)
            key = mx.random.uniform(shape=kv_shape, dtype=precision)
            value = mx.random.uniform(shape=kv_shape, dtype=precision)

            reference_output = mx.fast.scaled_dot_product_attention(
                q=query,
                k=key,
                v=value,
                scale=scale,
            )
            user_output = flash_attention(
                query,
                key,
                value,
                scale=scale,
            )
            mx.eval(user_output)  # so that any error will be caught here
            assert_allclose(user_output, reference_output, precision=mx.float16)


def test_flash_attention_cpu_small():
    attention_helper(mx.cpu, 6, 3, 2, 5, 3, 1)


def test_flash_attention_cpu():
    attention_helper(mx.cpu, 18, 6, 7, 5, 3, 10)


def test_flash_attention_cpu_large():
    attention_helper(mx.cpu, 28, 4, 16, 128, 16, 3)


def test_flash_attention_gpu_extra_small():
    attention_helper(mx.gpu, 1, 1, 5, 7, 4, 1)


def test_flash_attention_gpu_small():
    attention_helper(mx.gpu, 6, 3, 2, 5, 3, 1)


def test_flash_attention_gpu():
    attention_helper(mx.gpu, 18, 6, 7, 5, 3, 10)


def test_flash_attention_gpu_large():
    attention_helper(mx.gpu, 28, 4, 16, 128, 16, 3)
