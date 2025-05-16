import mlx.core as mx
import pytest
from mlx_lm.models import qwen2
import numpy as np

from .tiny_llm_base import *
from .utils import *

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
def test_qwen2_mlp_week_1_day_4_task_2(
    stream: mx.Stream, precision: np.dtype, dims: dict
):
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

        user_mlp = Qwen2MLP(
            dim=DIM, hidden_dim=HIDDEN_DIM, w_gate=w_gate, w_up=w_up, w_down=w_down
        )
        user_output = user_mlp(x)

        reference_mlp = qwen2.MLP(dim=DIM, hidden_dim=HIDDEN_DIM)
        reference_mlp.gate_proj.weight = w_gate
        reference_mlp.up_proj.weight = w_up
        reference_mlp.down_proj.weight = w_down
        reference_output = reference_mlp(x)

        assert_allclose(user_output, reference_output, precision)
