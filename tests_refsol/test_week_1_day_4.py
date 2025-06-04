import pytest
import mlx.core as mx
import mlx.nn as nn
from .tiny_llm_base import *
from .utils import *
from mlx_lm.models import qwen2


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rms_norm(
    stream: mx.Stream,
    precision: mx.Dtype,
):
    SIZE = 100
    SIZE_Y = 111
    with mx.stream(stream):
        for _ in range(100):
            data = mx.random.uniform(shape=(SIZE, SIZE_Y), dtype=precision)
            weight = mx.random.uniform(shape=(SIZE_Y,), dtype=precision)
            eps = mx.finfo(precision).eps
            reference_output = mx.fast.rms_norm(
                data,
                weight,
                eps=eps,
            )
            user_output = RMSNorm(SIZE_Y, weight, eps=eps)(data)
            assert_allclose(user_output, reference_output, precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_1_rms_norm_cast_to_float32(stream: mx.Stream):
    precision = mx.float16
    SIZE, SIZE_Y = 32, 64

    data = mx.random.uniform(-1000, 1000, shape=(SIZE, SIZE_Y), dtype=precision)
    weight = mx.random.uniform(-1000, 1000, shape=(SIZE_Y,), dtype=precision)
    eps = mx.finfo(precision).eps

    with mx.stream(stream):
        user_out = RMSNorm(SIZE_Y, weight, eps=eps)(data)
        ref_out = mx.fast.rms_norm(data, weight, eps=eps)

    assert_allclose(user_out, ref_out, precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_silu(stream: mx.Stream, precision: mx.Dtype):
    with mx.stream(stream):
        BATCH_SIZE = 10
        DIM = 10
        for _ in range(100):
            x = mx.random.uniform(shape=(BATCH_SIZE, DIM), dtype=precision)
            user_output = silu(x)
            reference_output = nn.silu(x)
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
def test_task_2_qwen_mlp(stream: mx.Stream, precision: mx.Dtype, dims: dict):
    BATCH_SIZE, SEQ_LEN, DIM, HIDDEN_DIM = (
        dims["batch_size"],
        dims["seq_len"],
        dims["dim"],
        dims["hidden_dim"],
    )

    with mx.stream(stream):
        x = mx.random.uniform(shape=(BATCH_SIZE, SEQ_LEN, DIM), dtype=precision)
        w_gate = mx.random.uniform(shape=(HIDDEN_DIM, DIM), dtype=precision)
        w_up = mx.random.uniform(shape=(HIDDEN_DIM, DIM), dtype=precision)
        w_down = mx.random.uniform(shape=(DIM, HIDDEN_DIM), dtype=precision)

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
