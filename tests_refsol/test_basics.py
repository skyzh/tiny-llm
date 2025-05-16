import pytest
import mlx.core as mx
import torch
from .tiny_llm_base import *
import numpy as np
from .utils import *


@pytest.mark.parametrize("target", ["torch", "mlx"])
@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_softmax(stream: mx.Stream, precision: np.dtype, target: str):
    with mx.stream(stream):
        BATCH_SIZE = 10
        DIM = 10
        for _ in range(100):
            x = np.random.rand(BATCH_SIZE, DIM).astype(precision)
            user_output = softmax(mx.array(x), axis=-1)
            if target == "torch":
                reference_output = torch.nn.functional.softmax(
                    torch.tensor(x, device=TORCH_DEVICE), dim=-1
                )
            else:
                reference_output = mx.softmax(mx.array(x), axis=-1)
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("target", ["torch", "mlx"])
@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_linear(stream: mx.Stream, precision: np.dtype, target: str):
    with mx.stream(stream):
        BATCH_SIZE = 10
        DIM_Y = 10
        DIM_X = 12
        for _ in range(100):
            x = np.random.rand(BATCH_SIZE, DIM_X).astype(precision)
            w = np.random.rand(DIM_Y, DIM_X).astype(precision)
            b = np.random.rand(DIM_Y).astype(precision)
            user_output = linear(mx.array(x), mx.array(w), mx.array(b))
            if target == "torch":
                reference_output = torch.nn.functional.linear(
                    torch.tensor(x, device=TORCH_DEVICE),
                    torch.tensor(w, device=TORCH_DEVICE),
                    torch.tensor(b, device=TORCH_DEVICE),
                )
            else:
                if precision == np.float16 and stream == mx.cpu:
                    # unsupported
                    break
                reference_output = mx.addmm(mx.array(b), mx.array(x), mx.array(w).T)
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("target", ["torch", "mlx"])
@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_silu(stream: mx.Stream, precision: np.dtype, target: str):
    with mx.stream(stream):
        BATCH_SIZE = 10
        DIM = 10
        for _ in range(100):
            x = np.random.rand(BATCH_SIZE, DIM).astype(precision)
            user_output = silu(mx.array(x))
            if target == "torch":
                reference_output = torch.nn.functional.silu(
                    torch.tensor(x, device=TORCH_DEVICE)
                )
            else:
                reference_output = silu(mx.array(x))
            assert_allclose(user_output, reference_output, precision=precision)
