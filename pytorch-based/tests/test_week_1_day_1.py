import pytest
import torch
import numpy as np
from .tiny_llm_base import *
from .utils import *
from .utils import assert_allclose
from .utils import softmax


@pytest.mark.parametrize("target", ["torch"])
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_softmax(precision: np.dtype, target: str):
    BATCH_SIZE = 10
    DIM = 10
    for _ in range(100):
        x = np.random.rand(BATCH_SIZE, DIM).astype(precision)
        user_output = softmax(torch.tensor(x, device=TORCH_DEVICE), axis=-1)
        reference_output = torch.nn.functional.softmax(
            torch.tensor(x, device=TORCH_DEVICE), dim=-1
        )
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"])
def test_task_1_simple_attention(precision: np.dtype, batch_dimension: int):
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

        query_t = torch.tensor(query, device=TORCH_DEVICE)
        key_t = torch.tensor(key, device=TORCH_DEVICE)
        value_t = torch.tensor(value, device=TORCH_DEVICE)

        user_output = scaled_dot_product_attention_simple(query_t, key_t, value_t)
        reference_output = torch.nn.functional.scaled_dot_product_attention(
            query_t, key_t, value_t, scale=1.0 / np.sqrt(DIM_D)
        )
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"])
def test_task_1_simple_attention_scale_mask(precision: np.dtype, batch_dimension: int):
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

        query_t = torch.tensor(query, device=TORCH_DEVICE)
        key_t = torch.tensor(key, device=TORCH_DEVICE)
        value_t = torch.tensor(value, device=TORCH_DEVICE)
        mask = torch.rand(*BATCH_SIZE, DIM_L, DIM_L, device=TORCH_DEVICE) 

        scale = 0.5

        user_output = scaled_dot_product_attention_simple(
            query_t, key_t, value_t, scale=scale, mask=mask.to(dtype=query_t.dtype)
        )

        reference_output = torch.nn.functional.scaled_dot_product_attention(
            query_t, key_t, value_t, attn_mask=mask, scale=scale
        )
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("target", ["torch"])
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_linear(precision: np.dtype, target: str):
    BATCH_SIZE = 10
    DIM_Y = 10
    DIM_X = 12
    for _ in range(100):
        x = np.random.rand(BATCH_SIZE, DIM_X).astype(precision)
        w = np.random.rand(DIM_Y, DIM_X).astype(precision)
        b = np.random.rand(DIM_Y).astype(precision)

        x_t = torch.tensor(x, device=TORCH_DEVICE)
        w_t = torch.tensor(w, device=TORCH_DEVICE)
        b_t = torch.tensor(b, device=TORCH_DEVICE)

        user_output = linear(x_t, w_t, b_t)
        reference_output = torch.nn.functional.linear(x_t, w_t, b_t)
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_simple_multi_head_attention(precision: np.dtype):
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

        query_t = torch.tensor(query, device=TORCH_DEVICE).transpose(0, 1)
        key_t = torch.tensor(key, device=TORCH_DEVICE).transpose(0, 1)
        value_t = torch.tensor(value, device=TORCH_DEVICE).transpose(0, 1)
        mask_t = torch.tensor(mask, device=TORCH_DEVICE)

        q_w = torch.tensor(q_proj_weight, device=TORCH_DEVICE)
        k_w = torch.tensor(k_proj_weight, device=TORCH_DEVICE)
        v_w = torch.tensor(v_proj_weight, device=TORCH_DEVICE)
        o_w = torch.tensor(out_proj_weight, device=TORCH_DEVICE)

        reference_output, _ = torch.nn.functional.multi_head_attention_forward(
            query_t, key_t, value_t,
            num_heads=H,
            q_proj_weight=q_w,
            k_proj_weight=k_w,
            v_proj_weight=v_w,
            out_proj_weight=o_w,
            embed_dim_to_check=H * D,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_bias=None,
            use_separate_proj_weight=True,
            attn_mask=mask_t,
        )
        reference_output = reference_output.transpose(0, 1)

        user_output = SimpleMultiHeadAttention(
            H * D, H, q_w, k_w, v_w, o_w
        )(
            torch.tensor(query, device=TORCH_DEVICE),
            torch.tensor(key, device=TORCH_DEVICE),
            torch.tensor(value, device=TORCH_DEVICE),
            mask=mask_t,
        )

        assert_allclose(user_output, reference_output, precision=precision)
