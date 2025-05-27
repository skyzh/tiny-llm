import pytest
import mlx.core as mx
import torch
from .tiny_llm_base import *
import numpy as np
from .utils import *
from mlx_lm.models import qwen2


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rms_norm(
    stream: mx.Stream,
    precision: np.dtype,
):
    SIZE = 100
    SIZE_Y = 111
    with mx.stream(stream):
        for _ in range(100):
            data = np.random.rand(SIZE, SIZE_Y).astype(precision)
            weight = np.random.rand(SIZE_Y).astype(precision)
            eps = np.finfo(precision).eps
            reference_output = mx.fast.rms_norm(
                mx.array(data),
                mx.array(weight),
                eps=eps,
            )
            user_output = RMSNorm(SIZE_Y, mx.array(weight), eps=eps)(mx.array(data))
            assert user_output.dtype == mx.array(data).dtype, (
                "Output dtype mismatch, perhaps you forgot to cast the output to the original dtype?"
            )
            assert_allclose(user_output, reference_output, precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_1_rms_norm_cast_to_float32(stream: mx.Stream):
    precision = np.float16
    SIZE, SIZE_Y = 32, 64

    data = (np.random.uniform(-1000, 1000, size=(SIZE, SIZE_Y))).astype(precision)
    weight = (np.random.uniform(-1000, 1000, size=(SIZE_Y,))).astype(precision)
    eps = np.finfo(precision).eps

    with mx.stream(stream):
        user_out = RMSNorm(SIZE_Y, mx.array(weight), eps=eps)(mx.array(data))
        ref_out = mx.fast.rms_norm(mx.array(data), mx.array(weight), eps=eps)

    assert_allclose(user_out, ref_out, precision)


@pytest.mark.parametrize("target", ["torch", "mlx"])
@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_silu(stream: mx.Stream, precision: np.dtype, target: str):
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


# Define different dimension parameters for testing
DIM_PARAMS = [
    {"batch_size": 1, "seq_len": 5, "dim": 4, "hidden_dim": 8, "id": "small_dims"},
    {"batch_size": 2, "seq_len": 16, "dim": 32, "hidden_dim": 64, "id": "large_dims"},
    {
        "batch_size": 1,
        "seq_len": 1,
        "dim": 128,
        "hidden_dim": 256,
        "id": "single_token",
    },
]
DIM_PARAMS_IDS = [d["id"] for d in DIM_PARAMS]


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("dims", DIM_PARAMS, ids=DIM_PARAMS_IDS)
def test_task_2_qwen_mlp(stream: mx.Stream, precision: np.dtype, dims: dict):
    BATCH_SIZE, SEQ_LEN, DIM, HIDDEN_DIM = (
        dims["batch_size"],
        dims["seq_len"],
        dims["dim"],
        dims["hidden_dim"],
    )

    with mx.stream(stream):
        mx_precision = np_type_to_mx_type(precision)
        x = mx.random.uniform(shape=(BATCH_SIZE, SEQ_LEN, DIM)).astype(mx_precision)
        w_gate = mx.random.uniform(shape=(HIDDEN_DIM, DIM)).astype(mx_precision)
        w_up = mx.random.uniform(shape=(HIDDEN_DIM, DIM)).astype(mx_precision)
        w_down = mx.random.uniform(shape=(DIM, HIDDEN_DIM)).astype(mx_precision)

        user_mlp = qwen2_week1.Qwen2MLP(
            dim=DIM, hidden_dim=HIDDEN_DIM, w_gate=w_gate, w_up=w_up, w_down=w_down
        )
        user_output = user_mlp(x)

        reference_mlp = qwen2.MLP(dim=DIM, hidden_dim=HIDDEN_DIM)
        reference_mlp.gate_proj.weight = w_gate
        reference_mlp.up_proj.weight = w_up
        reference_mlp.down_proj.weight = w_down
        reference_output = reference_mlp(x)

        assert_allclose(user_output, reference_output, precision)
