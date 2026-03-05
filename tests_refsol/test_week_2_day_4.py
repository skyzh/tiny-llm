import pytest
import time
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


def attention_helper(stream: mx.Stream, H_q, H, L, E, S, BATCH, mask_mode: str):
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
            if mask_mode == "no_mask":
                mask = None
            elif mask_mode == "mask":
                mask = mx.random.uniform(shape=mask_shape, dtype=precision)
            elif mask_mode == "causal":
                mask = "causal"
            else:
                raise ValueError(f"Unknown mask_mode: {mask_mode}")

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


def time_flash_attention(
    stream: mx.Stream,
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str,
    num_iters: int = 4,
) -> float:
    with mx.stream(stream):
        start = time.perf_counter()
        for _ in range(num_iters):
            output = flash_attention(query, key, value, scale=scale, mask=mask)
            mx.eval(output)
    return (time.perf_counter() - start) / num_iters


def median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def assert_causal_mask_faster_than_all_zero_mask(
    stream: mx.Stream,
    batch: int,
    h_q: int,
    h: int,
    l: int,
    s: int,
    e: int,
    scale: float = 0.9,
):
    precision = mx.float32
    q_shape = (batch, h_q, l, e)
    kv_shape = (batch, h, s, e)
    mask_shape = (batch, h_q, l, s)

    with mx.stream(stream):
        query = mx.random.uniform(shape=q_shape, dtype=precision)
        key = mx.random.uniform(shape=kv_shape, dtype=precision)
        value = mx.random.uniform(shape=kv_shape, dtype=precision)
        zero_mask = mx.zeros(shape=mask_shape, dtype=precision)

        for _ in range(3):
            mx.eval(flash_attention(query, key, value, scale=scale, mask="causal"))
            mx.eval(flash_attention(query, key, value, scale=scale, mask=zero_mask))

    causal_samples = []
    zero_mask_samples = []
    for round_idx in range(6):
        if round_idx % 2 == 0:
            causal_samples.append(
                time_flash_attention(
                    stream, query, key, value, scale=scale, mask="causal"
                )
            )
            zero_mask_samples.append(
                time_flash_attention(
                    stream, query, key, value, scale=scale, mask=zero_mask
                )
            )
        else:
            zero_mask_samples.append(
                time_flash_attention(
                    stream, query, key, value, scale=scale, mask=zero_mask
                )
            )
            causal_samples.append(
                time_flash_attention(
                    stream, query, key, value, scale=scale, mask="causal"
                )
            )

    causal_time = median(causal_samples)
    zero_mask_time = median(zero_mask_samples)
    assert causal_time < zero_mask_time, (
        "Expected causal mask to be faster than an all-zero mask, got "
        f"causal={causal_time:.6f}s and zero_mask={zero_mask_time:.6f}s."
    )


@pytest.mark.parametrize("mask_mode", ["no_mask", "mask", "causal"])
def test_task_2_flash_attention_cpu_small(mask_mode: str):
    attention_helper(mx.cpu, 6, 3, 2, 5, 3, 1, mask_mode)


@pytest.mark.parametrize("mask_mode", ["no_mask", "mask"])
def test_task_2_flash_attention_cpu(mask_mode: str):
    attention_helper(mx.cpu, 18, 6, 7, 5, 3, 10, mask_mode)


@pytest.mark.parametrize("mask_mode", ["no_mask", "mask", "causal"])
def test_task_2_flash_attention_cpu_large(mask_mode: str):
    attention_helper(mx.cpu, 28, 4, 16, 128, 16, 3, mask_mode)


def test_task_2_flash_attention_cpu_causal_mask_faster_than_all_zero_mask():
    assert_causal_mask_faster_than_all_zero_mask(
        stream=mx.cpu,
        batch=1,
        h_q=8,
        h=8,
        l=128,
        s=128,
        e=128,
    )


@pytest.mark.parametrize("mask_mode", ["no_mask", "mask"])
def test_task_3_flash_attention_gpu_extra_small(mask_mode: str):
    attention_helper(mx.gpu, 1, 1, 5, 7, 4, 1, mask_mode)


@pytest.mark.parametrize("mask_mode", ["no_mask", "mask", "causal"])
def test_task_3_flash_attention_gpu_small(mask_mode: str):
    attention_helper(mx.gpu, 6, 3, 2, 5, 3, 1, mask_mode)


@pytest.mark.parametrize("mask_mode", ["no_mask", "mask"])
def test_task_3_flash_attention_gpu(mask_mode: str):
    attention_helper(mx.gpu, 18, 6, 7, 5, 3, 10, mask_mode)


@pytest.mark.parametrize("mask_mode", ["no_mask", "mask", "causal"])
def test_task_3_flash_attention_gpu_large(mask_mode: str):
    attention_helper(mx.gpu, 28, 4, 16, 128, 16, 3, mask_mode)


def test_task_3_flash_attention_gpu_causal_mask_faster_than_all_zero_mask():
    assert_causal_mask_faster_than_all_zero_mask(
        stream=mx.gpu,
        batch=2,
        h_q=8,
        h=8,
        l=512,
        s=512,
        e=128,
    )
