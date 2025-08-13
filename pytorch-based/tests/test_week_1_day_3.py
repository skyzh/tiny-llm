import pytest
import torch
import numpy as np
from .tiny_llm_base import *
from .utils import *

#TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE="cpu"

def grouped_attention_helper( 
    stream,
    precision: np.dtype, 
    batch_dimension: int, 
    scale: float | None, 
    is_causal_mask: bool, 
): 
    H_q = 18 
    H = 6 
    L = 3 
    D = 5 
    S = 7 
    BATCH = 10 
    BATCH_2 = 2 
 
    dtype_map = { 
        np.float32: torch.float32, 
        np.float16: torch.float16, 
    } 
    torch_dtype = dtype_map[precision] 
 
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
        
        if not is_causal_mask:
            mask = np.random.choice([0.0, -10000.0], size=mask_shape, p=[0.8, 0.2]).astype(precision)
        else:
            mask = np.random.rand(*mask_shape).astype(precision)

        torch_query = torch.tensor(query, device=TORCH_DEVICE, dtype=torch_dtype) 
        torch_key = torch.tensor(key, device=TORCH_DEVICE, dtype=torch_dtype) 
        torch_value = torch.tensor(value, device=TORCH_DEVICE, dtype=torch_dtype) 
 
        head_dim = -3 
        if torch_query.shape[head_dim] != torch_key.shape[head_dim]: 
            assert torch_query.shape[head_dim] % torch_key.shape[head_dim] == 0 
            repeat_factor = torch_query.shape[head_dim] // torch_key.shape[head_dim] 
            torch_key = torch_key.repeat_interleave(repeat_factor, dim=head_dim) 
            torch_value = torch_value.repeat_interleave(repeat_factor, dim=head_dim)

        if is_causal_mask:
            if batch_dimension == 0:
                causal_mask_2d = causal_mask(L, S, torch_dtype)
                torch_mask = torch.tensor(causal_mask_2d, device=TORCH_DEVICE, dtype=torch_dtype)
                torch_mask = torch_mask.unsqueeze(0).expand(H_q, -1, -1)
            elif batch_dimension == 1:
                causal_mask_2d = causal_mask(L, S, torch_dtype)
                torch_mask = torch.tensor(causal_mask_2d, device=TORCH_DEVICE, dtype=torch_dtype)
                torch_mask = torch_mask.unsqueeze(0).unsqueeze(0).expand(BATCH, H_q, -1, -1)
            elif batch_dimension == 2:
                causal_mask_2d = causal_mask(L, S, torch_dtype)
                torch_mask = torch.tensor(causal_mask_2d, device=TORCH_DEVICE, dtype=torch_dtype)
                torch_mask = torch_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(BATCH_2, BATCH, H_q, -1, -1)
        else:
            torch_mask = torch.tensor(mask, device=TORCH_DEVICE, dtype=torch_dtype)

        expected_mask_shape = torch_query.shape[:-1] + (torch_key.shape[-2],)
        assert torch_mask.shape == expected_mask_shape, \
            f"Mask shape mismatch: {torch_mask.shape} vs expected {expected_mask_shape}"
        
        print("query shape:", torch_query.shape) 
        print("key shape:", torch_key.shape) 
        print("mask shape:", torch_mask.shape) 

        if is_causal_mask:
            L, S = torch_query.shape[-2], torch_key.shape[-2]
            reference_output = torch.nn.functional.scaled_dot_product_attention(
                torch_query,
                torch_key,
                torch_value,
                attn_mask=causal_mask(L, S, dtype=torch_query.dtype).to(torch_query.device),
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )

        else:
            reference_output = torch.nn.functional.scaled_dot_product_attention( 
                torch_query, 
                torch_key, 
                torch_value, 
                attn_mask=torch_mask, 
                dropout_p=0.0, 
                is_causal=False,
                scale=scale,
            )
 
        user_output = scaled_dot_product_attention_grouped( 
            torch_query, torch_key, torch_value, scale=scale, mask=torch_mask 
        ) 
 
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"])
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_1_grouped_attention(
    stream, precision: np.dtype, batch_dimension: int, scale: float | None
):
    grouped_attention_helper(stream, precision, batch_dimension, scale, False)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_2_mask_only_same_dim(stream):
    L = 3
    S = 3
    user_output = causal_mask(L, S, torch.float32)
    expected = torch.tensor([
        [0, -np.inf, -np.inf],
        [0, 0, -np.inf],
        [0, 0, 0],
    ], dtype=torch.float32)
    assert_allclose(user_output, expected, precision=np.float32)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_2_mask_only_different_dim(stream):
    L = 3
    S = 5
    user_output = causal_mask(L, S, torch.float32)
    expected = torch.tensor([
        [0, 0, 0, -np.inf, -np.inf],
        [0, 0, 0, 0, -np.inf],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    assert_allclose(user_output, expected, precision=np.float32)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"])
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_2_grouped_attention_causal_mask(
    stream, precision: np.dtype, batch_dimension: int, scale: float | None
):
    grouped_attention_helper(stream, precision, batch_dimension, scale, True)


def test_task_3_qwen2_grouped_query_attention():
    pass
