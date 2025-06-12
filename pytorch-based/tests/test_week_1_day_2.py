import pytest
import torch
import numpy as np
from .tiny_llm_base import *
from .utils import *

def rope_reference(x, dims, traditional, base, scale=1.0, offset=0):
    B, H, L, D = x.shape
    assert D == dims, f"D mismatch: {D} != {dims}"
    assert D % 2 == 0, "Head dim must be even"

    theta = 1.0 / (base ** (torch.arange(0, dims, 2, device=x.device, dtype=x.dtype) / dims))
    position = torch.arange(offset, offset + L, device=x.device, dtype=x.dtype)
    freqs = torch.outer(position, theta) * scale

    cos = freqs.cos().unsqueeze(0).unsqueeze(2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(2)

    x = x.permute(0, 2, 1, 3)

    if traditional:
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        return (x * cos + rotate_half(x) * sin).permute(0, 2, 1, 3)
    else:
        x1 = x[..., :D // 2]
        x2 = x[..., D // 2:]

        cos = cos.expand(-1, -1, H, -1)
        sin = sin.expand(-1, -1, H, -1)

        out1 = x1 * cos + x2 * sin
        out2 = -x1 * sin + x2 * cos

        out = torch.cat([out1, out2], dim=-1)
        return out.permute(0, 2, 1, 3)
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def rope_helper(
    stream,
    traditional: bool,
    precision: np.dtype,
    with_offset: bool,
):
    BATCH_SIZE = 1
    NUM_HEADS = 8
    HEAD_DIM = 4
    MAX_SEQ_LEN = 20
    SEQ_LEN = 10
    BASE = 10000.0

    dtype = torch.float32 if precision == np.float32 else torch.float16

    for _ in range(100):
        user_layer = RoPE(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=traditional)
        x = np.random.rand(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM).astype(precision)
        x_torch = torch.tensor(x, dtype=dtype)

        if with_offset:
            input_pos = np.random.randint(0, MAX_SEQ_LEN - SEQ_LEN)
            input_pos_user = slice(input_pos, input_pos + SEQ_LEN)
        else:
            input_pos = None
            input_pos_user = None





        reference_output = rope_reference(
            torch.tensor(x).permute(0, 2, 1, 3).to(dtype),
            dims=HEAD_DIM,
            traditional=traditional,
            base=BASE,
            scale=1.0,
            offset=input_pos or 0,
        ).permute(0, 2, 1, 3)

        user_output = user_layer(x_torch, input_pos_user)

        assert_allclose(
            user_output.detach().cpu().numpy(),
            reference_output.detach().cpu().numpy(),
            precision,
            atol=5e-6 if precision == np.float32 else 1e-3,
        )


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("with_offset", [True, False], ids=["with_offset", "without_offset"])
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rope_mlx_traditional(
    stream, with_offset: bool, precision: np.dtype
):
    rope_helper(stream, True, precision, with_offset)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("with_offset", [True, False], ids=["with_offset", "without_offset"])
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_rope_mlx_non_traditional(
    stream, with_offset: bool, precision: np.dtype
):
    rope_helper(stream, False, precision, with_offset)
