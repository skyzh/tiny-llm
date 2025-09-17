import mlx.core as mx
import numpy as np
import pytest
from .tiny_llm_base import *
from .utils import *


def rope_helper(stream: mx.Stream, traditional: bool, precision: mx.Dtype):
    BATCH_SIZE = 16
    NUM_HEADS = 8
    HEAD_DIM = 4
    MAX_SEQ_LEN = 14
    SEQ_LEN = 9
    BASE = 10000
    with mx.stream(stream):
        for _ in range(100):
            user_layer = RoPE(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=traditional)
            x = mx.random.uniform(
                shape=(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=precision
            )

            input_pos = np.random.randint(0, MAX_SEQ_LEN - SEQ_LEN, size=BATCH_SIZE)
            input_pos_mx = mx.array(input_pos, dtype=mx.int32)
            input_pos_user = [slice(i, i + SEQ_LEN) for i in input_pos]

            reference_output = mx.fast.rope(
                x.transpose(0, 2, 1, 3),
                dims=HEAD_DIM,
                traditional=traditional,
                base=BASE,
                scale=1.0,
                offset=input_pos_mx,
            ).transpose(0, 2, 1, 3)
            user_output = user_layer(x, input_pos_user)
            assert_allclose(
                user_output,
                reference_output,
                precision,
                atol=5e-6 if precision == mx.float32 else 1e-3,
            )


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("traditional", [False, True], ids=["default", "traditional"])
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rope_multiple_offsets(
    stream: mx.Stream, traditional: bool, precision: mx.Dtype
):
    rope_helper(stream, traditional, precision)


def attention_helper(
    stream: mx.Stream, H_q, H, L, E, S, BATCH, use_flash_attention: bool = False
):
    precision = mx.float32
    with mx.stream(stream):
        q_shape = (BATCH, H_q, L, E)
        kv_shape = (BATCH, H, S, E)
        scale = 0.8
        for _ in range(100):
            query = mx.random.uniform(shape=q_shape, dtype=precision)
            key = mx.random.uniform(shape=kv_shape, dtype=precision)
            value = mx.random.uniform(shape=kv_shape, dtype=precision)
            mask = mx.random.uniform(shape=(BATCH, 1, L, S), dtype=precision)

            reference_output_1 = mx.fast.scaled_dot_product_attention(
                q=query,
                k=key,
                v=value,
                scale=scale,
                mask=mask,
            )
            reference_output_2 = mx.fast.scaled_dot_product_attention(
                q=query,
                k=key,
                v=value,
                scale=scale,
            )
            if use_flash_attention:
                user_output_1 = flash_attention(
                    query,
                    key,
                    value,
                    scale=scale,
                    mask=mask,
                )
                user_output_2 = flash_attention(
                    query,
                    key,
                    value,
                    scale=scale,
                )
            else:
                user_output_1 = scaled_dot_product_attention_grouped(
                    query,
                    key,
                    value,
                    scale=scale,
                    mask=mask,
                )
                user_output_2 = scaled_dot_product_attention_grouped(
                    query,
                    key,
                    value,
                    scale=scale,
                )
            mx.eval(user_output_1)
            mx.eval(user_output_2)
            assert_allclose(
                user_output_2,
                reference_output_2,
                precision=mx.float16,
                message="no mask",
            )
            assert_allclose(
                user_output_1,
                reference_output_1,
                precision=mx.float16,
                message="with mask",
            )


def test_task_1_flash_attention_with_mask_cpu_small():
    attention_helper(mx.cpu, 6, 3, 2, 5, 3, 1, use_flash_attention=True)


def test_task_1_flash_attention_with_mask_cpu():
    attention_helper(mx.cpu, 18, 6, 7, 5, 3, 10, use_flash_attention=True)


def test_task_1_flash_attention_with_mask_cpu_large():
    attention_helper(mx.cpu, 28, 4, 16, 128, 16, 3, use_flash_attention=True)


def test_task_1_flash_attention_with_mask_gpu_extra_small():
    attention_helper(mx.gpu, 1, 1, 5, 7, 4, 1, use_flash_attention=True)


def test_task_1_flash_attention_with_mask_gpu_small():
    attention_helper(mx.gpu, 6, 3, 2, 5, 3, 1, use_flash_attention=True)


def test_task_1_flash_attention_with_mask_gpu():
    attention_helper(mx.gpu, 18, 6, 7, 5, 3, 10, use_flash_attention=True)


def test_task_1_flash_attention_with_mask_gpu_large():
    attention_helper(mx.gpu, 28, 4, 16, 128, 16, 3, use_flash_attention=True)


def test_task_1_attention_with_mask_cpu_small():
    attention_helper(mx.cpu, 6, 3, 2, 5, 3, 1, use_flash_attention=False)


def test_task_1_attention_with_mask_cpu():
    attention_helper(mx.cpu, 18, 6, 7, 5, 3, 10, use_flash_attention=False)


def test_task_1_attention_with_mask_cpu_large():
    attention_helper(mx.cpu, 28, 4, 16, 128, 16, 3, use_flash_attention=False)


def test_task_1_attention_with_mask_gpu_extra_small():
    attention_helper(mx.gpu, 1, 1, 5, 7, 4, 1, use_flash_attention=False)


def test_task_1_attention_with_mask_gpu_small():
    attention_helper(mx.gpu, 6, 3, 2, 5, 3, 1, use_flash_attention=False)


def test_task_1_attention_with_mask_gpu():
    attention_helper(mx.gpu, 18, 6, 7, 5, 3, 10, use_flash_attention=False)


def test_task_1_attention_with_mask_gpu_large():
    attention_helper(mx.gpu, 28, 4, 16, 128, 16, 3, use_flash_attention=False)
