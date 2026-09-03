import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .tiny_llm_base import *
from .utils import *
from mlx_lm.models import qwen3


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
@pytest.mark.parametrize("precision", [mx.float16, mx.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize(
    "leading_shape", [(2, 3), (2, 1, 3)], ids=["two_leading", "three_leading"]
)
def test_task_1_rms_norm_leading_dimensions_and_dtype(
    stream: mx.Stream,
    precision: mx.Dtype,
    leading_shape: tuple[int, ...],
):
    dim = 5
    eps = 3e-4
    shape = (*leading_shape, dim)
    data = mx.array(np.linspace(-3.5, 2.75, np.prod(shape)).reshape(shape)).astype(
        precision
    )
    weight = mx.array(np.linspace(0.5, 1.5, dim)).astype(precision)

    data_f32 = np.array(data.astype(mx.float32))
    normalized_f32 = data_f32 / np.sqrt(
        np.mean(np.square(data_f32), axis=-1, keepdims=True) + eps
    )
    expected = mx.array(normalized_f32).astype(precision) * weight

    with mx.stream(stream):
        output = RMSNorm(dim, weight, eps=eps)(data)

    assert output.shape == shape
    assert output.dtype == precision
    assert_allclose(output, expected, precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize(
    "precision",
    [*PRECISIONS, mx.bfloat16],
    ids=[*PRECISION_IDS, "bf16"],
)
def test_task_2_silu(stream: mx.Stream, precision: mx.Dtype):
    with mx.stream(stream):
        BATCH_SIZE = 10
        DIM = 10
        for _ in range(100):
            x = mx.random.uniform(
                low=-20,
                high=20,
                shape=(BATCH_SIZE, DIM),
                dtype=precision,
            )
            user_output = silu(x)
            reference_output = nn.silu(x)
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_2_silu_low_precision_tails(stream: mx.Stream):
    precision = mx.float16
    tail = -12.0
    data = mx.array([tail, -4.0, 0.0, 4.0, -tail]).astype(precision)
    data_f32 = np.array(data.astype(mx.float32))
    sigmoid = np.where(
        data_f32 < 0,
        np.exp(data_f32) / (1 + np.exp(data_f32)),
        1 / (1 + np.exp(-data_f32)),
    )
    expected = mx.array(data_f32 * sigmoid).astype(precision)

    with mx.stream(stream):
        output = silu(data)

    output_f32 = np.array(output.astype(mx.float32))
    assert output.dtype == precision
    assert np.all(np.isfinite(output_f32))
    assert output_f32[0] != 0
    assert_allclose(output, expected, precision=precision)


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

        user_mlp = qwen3_week1.Qwen3MLP(
            dim=DIM, hidden_dim=HIDDEN_DIM, w_gate=w_gate, w_up=w_up, w_down=w_down
        )
        user_output = user_mlp(x)

        reference_mlp = qwen3.MLP(dim=DIM, hidden_dim=HIDDEN_DIM)
        reference_mlp.gate_proj.weight = w_gate
        reference_mlp.up_proj.weight = w_up
        reference_mlp.down_proj.weight = w_down
        reference_output = reference_mlp(x)

        assert_allclose(user_output, reference_output, precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_2_qwen_mlp_leading_dimensions_bfloat16(
    stream: mx.Stream,
):
    precision = mx.bfloat16
    dim = 4
    hidden_dim = 6
    shape = (2, 1, 3, dim)
    data = mx.array(np.linspace(-0.5, 0.5, np.prod(shape)).reshape(shape)).astype(
        precision
    )
    w_gate = mx.array(
        np.linspace(-0.4, 0.3, hidden_dim * dim).reshape(hidden_dim, dim)
    ).astype(precision)
    w_up = mx.array(
        np.linspace(0.35, -0.25, hidden_dim * dim).reshape(hidden_dim, dim)
    ).astype(precision)
    w_down = mx.array(
        np.linspace(-0.3, 0.45, dim * hidden_dim).reshape(dim, hidden_dim)
    ).astype(precision)

    data_f32 = np.array(data.astype(mx.float32))
    gate = data_f32 @ np.array(w_gate.astype(mx.float32)).T
    up = data_f32 @ np.array(w_up.astype(mx.float32)).T
    sigmoid = np.where(
        gate < 0,
        np.exp(gate) / (1 + np.exp(gate)),
        1 / (1 + np.exp(-gate)),
    )
    expected = (gate * sigmoid * up) @ np.array(w_down.astype(mx.float32)).T

    with mx.stream(stream):
        output = qwen3_week1.Qwen3MLP(
            dim=dim,
            hidden_dim=hidden_dim,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )(data)

    output_f32 = np.array(output.astype(mx.float32))
    assert output.shape == shape
    assert output.dtype == precision
    assert np.all(np.isfinite(output_f32))
    assert_allclose(
        output,
        mx.array(expected).astype(precision),
        precision=precision,
        rtol=0.08,
        atol=2e-3,
    )
