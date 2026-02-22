import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


def attention_helper(stream: mx.Stream, H_q, H, L, E, S, BATCH, with_mask: bool):
    precision = mx.float32
    with mx.stream(stream):
        q_shape = (BATCH, H_q, L, E)
        kv_shape = (BATCH, H, S, E)
        mask_shape = (BATCH, H_q, L, S)
        scale = 0.9
        for _ in range(100):
            query = mx.random.uniform(shape=q_shape, dtype=precision)
            key = mx.random.uniform(shape=kv_shape, dtype=precision)
            value = mx.random.uniform(shape=kv_shape, dtype=precision)
            mask = mx.random.uniform(shape=mask_shape, dtype=precision) if with_mask else None

            reference_output = mx.fast.scaled_dot_product_attention(
                q=query,
                k=key,
                v=value,
                scale=scale,
                mask=mask,
            )
            user_output = flash_attention(
                query,
                key,
                value,
                scale=scale,
                mask=mask,
            )
            mx.eval(user_output)  # so that any error will be caught here
            assert_allclose(user_output, reference_output, precision=mx.float16)


@pytest.mark.parametrize("with_mask", [False, True], ids=["no_mask", "mask"])
def test_task_2_flash_attention_cpu_small(with_mask: bool):
    attention_helper(mx.cpu, 6, 3, 2, 5, 3, 1, with_mask)


@pytest.mark.parametrize("with_mask", [False, True], ids=["no_mask", "mask"])
def test_task_2_flash_attention_cpu(with_mask: bool):
    attention_helper(mx.cpu, 18, 6, 7, 5, 3, 10, with_mask)


@pytest.mark.parametrize("with_mask", [False, True], ids=["no_mask", "mask"])
def test_task_2_flash_attention_cpu_large(with_mask: bool):
    attention_helper(mx.cpu, 28, 4, 16, 128, 16, 3, with_mask)


@pytest.mark.parametrize("with_mask", [False, True], ids=["no_mask", "mask"])
def test_task_3_flash_attention_gpu_extra_small(with_mask: bool):
    attention_helper(mx.gpu, 1, 1, 5, 7, 4, 1, with_mask)


@pytest.mark.parametrize("with_mask", [False, True], ids=["no_mask", "mask"])
def test_task_3_flash_attention_gpu_small(with_mask: bool):
    attention_helper(mx.gpu, 6, 3, 2, 5, 3, 1, with_mask)


@pytest.mark.parametrize("with_mask", [False, True], ids=["no_mask", "mask"])
def test_task_3_flash_attention_gpu(with_mask: bool):
    attention_helper(mx.gpu, 18, 6, 7, 5, 3, 10, with_mask)


@pytest.mark.parametrize("with_mask", [False, True], ids=["no_mask", "mask"])
def test_task_3_flash_attention_gpu_large(with_mask: bool):
    attention_helper(mx.gpu, 28, 4, 16, 128, 16, 3, with_mask)
