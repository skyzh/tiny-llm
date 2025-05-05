import pytest
import mlx.core as mx
import torch
from .tiny_llm_base import *
import numpy as np
from .utils import *


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_attention_week_1_day_1_task_1(stream: mx.Stream, precision: np.dtype):
    with mx.stream(stream):
        BATCH_SIZE = 3
        DIM_N = 4
        DIM_M = 5
        for _ in range(100):
            query = np.random.rand(BATCH_SIZE, DIM_N, DIM_M).astype(precision)
            key = np.random.rand(BATCH_SIZE, DIM_N, DIM_M).astype(precision)
            value = np.random.rand(BATCH_SIZE, DIM_N, DIM_M).astype(precision)
            reference_output = torch.nn.functional.scaled_dot_product_attention(
                torch.tensor(query, device=TORCH_DEVICE),
                torch.tensor(key, device=TORCH_DEVICE),
                torch.tensor(value, device=TORCH_DEVICE),
            )
            user_output = scaled_dot_product_attention(
                mx.array(query),
                mx.array(key),
                mx.array(value),
            )
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "qkv_shape", [True, False], ids=["with_seq_len", "without_seq_len"]
)
def test_attention_with_mask_week_1_day_1_task_1(
    stream: mx.Stream, precision: np.dtype, qkv_shape: bool
):
    with mx.stream(stream):
        BATCH_SIZE = 3
        SEQ_LEN = 10
        H = 4
        D = 5
        if qkv_shape:
            qkv_shape = (BATCH_SIZE, H, SEQ_LEN, D)
            mask_shape = (BATCH_SIZE, H, SEQ_LEN, SEQ_LEN)
        else:
            qkv_shape = (BATCH_SIZE, H, SEQ_LEN, D)
            mask_shape = (BATCH_SIZE, H, SEQ_LEN, SEQ_LEN)
        for _ in range(100):
            query = np.random.rand(*qkv_shape).astype(precision)
            key = np.random.rand(*qkv_shape).astype(precision)
            value = np.random.rand(*qkv_shape).astype(precision)
            scale = 0.8
            mask = np.random.rand(*mask_shape).astype(precision)
            reference_output = torch.nn.functional.scaled_dot_product_attention(
                torch.tensor(query, device=TORCH_DEVICE),
                torch.tensor(key, device=TORCH_DEVICE),
                torch.tensor(value, device=TORCH_DEVICE),
                scale=scale,
                attn_mask=torch.tensor(mask, device=TORCH_DEVICE),
            )
            user_output = scaled_dot_product_attention(
                mx.array(query),
                mx.array(key),
                mx.array(value),
                scale=scale,
                mask=mx.array(mask),
            )
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_multi_head_attention_week_1_day_1_task_2(
    stream: mx.Stream, precision: np.dtype
):
    with mx.stream(stream):
        SEQ_LEN = 11
        D = 9
        H = 3
        BATCH_SIZE = 10
        for _ in range(100):
            query = np.random.rand(BATCH_SIZE, SEQ_LEN, H * D).astype(precision)
            key = np.random.rand(BATCH_SIZE, SEQ_LEN, H * D).astype(precision)
            value = np.random.rand(BATCH_SIZE, SEQ_LEN, H * D).astype(precision)
            q_proj_weight = np.random.rand(H * D, H * D).astype(precision)
            k_proj_weight = np.random.rand(H * D, H * D).astype(precision)
            v_proj_weight = np.random.rand(H * D, H * D).astype(precision)
            out_proj_weight = np.random.rand(H * D, H * D).astype(precision)
            mask = np.random.rand(SEQ_LEN, SEQ_LEN).astype(precision)
            reference_output, _ = torch.nn.functional.multi_head_attention_forward(
                torch.tensor(query, device=TORCH_DEVICE).transpose(0, 1),
                torch.tensor(key, device=TORCH_DEVICE).transpose(0, 1),
                torch.tensor(value, device=TORCH_DEVICE).transpose(0, 1),
                num_heads=H,
                q_proj_weight=torch.tensor(q_proj_weight, device=TORCH_DEVICE),
                k_proj_weight=torch.tensor(k_proj_weight, device=TORCH_DEVICE),
                v_proj_weight=torch.tensor(v_proj_weight, device=TORCH_DEVICE),
                out_proj_weight=torch.tensor(out_proj_weight, device=TORCH_DEVICE),
                embed_dim_to_check=H * D,
                in_proj_weight=None,
                in_proj_bias=None,
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0.0,
                out_proj_bias=None,
                use_separate_proj_weight=True,
                attn_mask=torch.tensor(mask, device=TORCH_DEVICE),
            )
            reference_output = reference_output.transpose(0, 1)
            user_output = MultiHeadAttention(
                H * D,
                H,
                mx.array(q_proj_weight),
                mx.array(k_proj_weight),
                mx.array(v_proj_weight),
                mx.array(out_proj_weight),
            )(
                mx.array(query),
                mx.array(key),
                mx.array(value),
                mask=mx.array(mask),
            )
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_attention_grouped_week_1_day_3_task_1(
    stream: mx.Stream, precision: np.dtype, batch_dimension: int, scale: float | None
):
    with mx.stream(stream):
        H_q = 18
        H = 6
        L = 7
        D = 5
        S = 3
        BATCH = 10
        BATCH_2 = 2
        if batch_dimension == 0:
            q_shape = (H_q, L, D)
            kv_shape = (H, S, D)
            mask_shape = (H_q, L, S)
        elif batch_dimension == 1:
            q_shape = (BATCH, H_q, L, D)
            kv_shape = (BATCH, H, S, D)
            mask_shape = (BATCH, H_q, L, S)
        elif batch_dimension == 2:
            q_shape = (BATCH_2, BATCH, H_q, L, D)
            kv_shape = (BATCH_2, BATCH, H, S, D)
            mask_shape = (BATCH_2, BATCH, H_q, L, S)
        for _ in range(100):
            query = np.random.rand(*q_shape).astype(precision)
            key = np.random.rand(*kv_shape).astype(precision)
            value = np.random.rand(*kv_shape).astype(precision)
            mask = np.random.rand(*mask_shape).astype(precision)
            reference_output = torch.nn.functional.scaled_dot_product_attention(
                torch.tensor(query, device=TORCH_DEVICE),
                torch.tensor(key, device=TORCH_DEVICE),
                torch.tensor(value, device=TORCH_DEVICE),
                scale=scale,
                attn_mask=torch.tensor(mask, device=TORCH_DEVICE),
                enable_gqa=True,
            )
            user_output = scaled_dot_product_attention_grouped(
                mx.array(query),
                mx.array(key),
                mx.array(value),
                scale=scale,
                mask=mx.array(mask),
            )
            assert_allclose(user_output, reference_output, precision=precision)
