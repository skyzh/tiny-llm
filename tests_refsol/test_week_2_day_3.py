import pytest
import mlx.core as mx
from .tiny_llm_base import *
import numpy as np
from .utils import *


def attention_helper(stream: mx.Stream, H_q, H, L, E, S, BATCH):
    precision = np.float32
    with mx.stream(stream):
        q_shape = (BATCH, H_q, L, E)
        kv_shape = (BATCH, H, S, E)
        scale = 1.0
        for _ in range(100):
            query = np.random.rand(*q_shape).astype(precision)
            key = np.random.rand(*kv_shape).astype(precision)
            value = np.random.rand(*kv_shape).astype(precision)
            reference_output = torch.nn.functional.scaled_dot_product_attention(
                torch.tensor(query, device=TORCH_DEVICE),
                torch.tensor(key, device=TORCH_DEVICE),
                torch.tensor(value, device=TORCH_DEVICE),
                scale=scale,
                enable_gqa=True,
            )
            user_output = flash_attention(
                mx.array(query),
                mx.array(key),
                mx.array(value),
                scale=scale,
            )
            mx.eval(user_output)  # so that any error will be caught here
            assert_allclose(user_output, reference_output, precision=precision)


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
