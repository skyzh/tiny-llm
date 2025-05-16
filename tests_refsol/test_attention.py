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
