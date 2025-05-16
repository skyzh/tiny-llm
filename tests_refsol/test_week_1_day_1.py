import pytest
import mlx.core as mx
import torch
from .tiny_llm_base import *
import numpy as np
from .utils import *


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
def test_task_1_simple_attention(
    stream: mx.Stream, precision: np.dtype, batch_dimension: int
):
    """
    Test if `scaled_dot_product_attention_simple` can process Q/K/V correctly.
    We assume Q/K/V are of the same dimensions and test different batch dimensions.
    """
    with mx.stream(stream):
        if batch_dimension == 0:
            BATCH_SIZE = ()
        elif batch_dimension == 1:
            BATCH_SIZE = (2, 3)
        elif batch_dimension == 2:
            BATCH_SIZE = (2, 3, 3)
        DIM_L = 4
        DIM_D = 5
        for _ in range(100):
            query = np.random.rand(*BATCH_SIZE, DIM_L, DIM_D).astype(precision)
            key = np.random.rand(*BATCH_SIZE, DIM_L, DIM_D).astype(precision)
            value = np.random.rand(*BATCH_SIZE, DIM_L, DIM_D).astype(precision)
            reference_output = mx.fast.scaled_dot_product_attention(
                q=mx.array(query).reshape(1, -1, DIM_L, DIM_D),
                k=mx.array(key).reshape(1, -1, DIM_L, DIM_D),
                v=mx.array(value).reshape(1, -1, DIM_L, DIM_D),
                scale=mx.rsqrt(DIM_D),
            ).reshape(*BATCH_SIZE, DIM_L, DIM_D)
            user_output = scaled_dot_product_attention_simple(
                mx.array(query),
                mx.array(key),
                mx.array(value),
            )
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
def test_task_1_simple_attention_scale_mask(
    stream: mx.Stream, precision: np.dtype, batch_dimension: int
):
    """
    Test if `scaled_dot_product_attention_simple` can process scale and mask correctly.
    """
    with mx.stream(stream):
        if batch_dimension == 0:
            BATCH_SIZE = ()
        elif batch_dimension == 1:
            BATCH_SIZE = (2, 3)
        elif batch_dimension == 2:
            BATCH_SIZE = (2, 3, 3)
        DIM_L = 4
        DIM_D = 5
        for _ in range(100):
            query = np.random.rand(*BATCH_SIZE, DIM_L, DIM_D).astype(precision)
            key = np.random.rand(*BATCH_SIZE, DIM_L, DIM_D).astype(precision)
            value = np.random.rand(*BATCH_SIZE, DIM_L, DIM_D).astype(precision)
            mask = np.random.rand(*BATCH_SIZE, DIM_L, DIM_L).astype(precision)
            scale = 0.5
            reference_output = mx.fast.scaled_dot_product_attention(
                q=mx.array(query).reshape(1, -1, DIM_L, DIM_D),
                k=mx.array(key).reshape(1, -1, DIM_L, DIM_D),
                v=mx.array(value).reshape(1, -1, DIM_L, DIM_D),
                scale=scale,
                mask=mx.array(mask).reshape(1, -1, DIM_L, DIM_L),
            ).reshape(*BATCH_SIZE, DIM_L, DIM_D)
            user_output = scaled_dot_product_attention_simple(
                mx.array(query),
                mx.array(key),
                mx.array(value),
                scale=scale,
                mask=mx.array(mask),
            )
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_simple_multi_head_attention(stream: mx.Stream, precision: np.dtype):
    """
    Test if `MultiHeadAttention` can process everything correctly. We assume Q/K/V are of the same dimensions.
    """
    with mx.stream(stream):
        L = 11
        D = 9
        H = 3
        BATCH_SIZE = 10
        for _ in range(100):
            query = np.random.rand(BATCH_SIZE, L, H * D).astype(precision)
            key = np.random.rand(BATCH_SIZE, L, H * D).astype(precision)
            value = np.random.rand(BATCH_SIZE, L, H * D).astype(precision)
            q_proj_weight = np.random.rand(H * D, H * D).astype(precision)
            k_proj_weight = np.random.rand(H * D, H * D).astype(precision)
            v_proj_weight = np.random.rand(H * D, H * D).astype(precision)
            out_proj_weight = np.random.rand(H * D, H * D).astype(precision)
            mask = np.random.rand(L, L).astype(precision)
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
