import pytest
import mlx.core as mx
import torch
from .tiny_llm_base import *
import numpy as np
from .utils import *
import torchtune


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize(
    "with_offset", [True, False], ids=["with_offset", "without_offset"]
)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_rope_torch_week_1_day_2_task_1(
    stream: mx.Stream, with_offset: bool, precision: np.dtype
):
    BATCH_SIZE = 1
    NUM_HEADS = 8
    HEAD_DIM = 4
    MAX_SEQ_LEN = 20
    SEQ_LEN = 10
    BASE = 10000.0

    with mx.stream(stream):
        for _ in range(100):
            reference_layer = (
                torchtune.modules.position_embeddings.RotaryPositionalEmbeddings(
                    HEAD_DIM,
                    MAX_SEQ_LEN,
                    BASE,
                )
            )
            user_layer = RoPE(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=True)
            x = np.random.rand(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM).astype(
                precision
            )

            if with_offset:
                input_pos = np.random.randint(0, MAX_SEQ_LEN - SEQ_LEN)
                input_pos_user = slice(input_pos, input_pos + SEQ_LEN)
                input_pos_torch = torch.tensor(
                    [i for i in range(input_pos, input_pos + SEQ_LEN)],
                    device=TORCH_DEVICE,
                    dtype=torch.int32,
                )
            else:
                input_pos = None
                input_pos_user = None
                input_pos_torch = None

            reference_output = reference_layer.forward(
                torch.tensor(x, device=TORCH_DEVICE), input_pos=input_pos_torch
            )
            user_output = user_layer(mx.array(x), input_pos_user)
            assert_allclose(user_output, reference_output, precision, atol=1e-6)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize(
    "with_offset", [True, False], ids=["with_offset", "without_offset"]
)
@pytest.mark.parametrize(
    "traditional",
    [True, False],
    ids=["traditional_week_1_day_2_task_1", "non_traditional_week_1_day_2_task_2"],
)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_rope_mlx(
    stream: mx.Stream, with_offset: bool, traditional: bool, precision: np.dtype
):
    BATCH_SIZE = 1
    NUM_HEADS = 8
    HEAD_DIM = 4
    MAX_SEQ_LEN = 20
    SEQ_LEN = 10
    BASE = 10000.0
    with mx.stream(stream):
        for _ in range(100):
            user_layer = RoPE(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=traditional)
            x = np.random.rand(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM).astype(
                precision
            )

            if with_offset:
                input_pos = np.random.randint(0, MAX_SEQ_LEN - SEQ_LEN)
                input_pos_mx = input_pos
                input_pos_user = slice(input_pos, input_pos + SEQ_LEN)
            else:
                input_pos = None
                input_pos_mx = None
                input_pos_user = None

            reference_output = mx.fast.rope(
                mx.array(x).transpose(0, 2, 1, 3),
                dims=HEAD_DIM,
                traditional=traditional,
                base=BASE,
                scale=1.0,
                offset=input_pos_mx or 0,
            ).transpose(0, 2, 1, 3)
            user_output = user_layer(mx.array(x), input_pos_user)
            assert_allclose(
                user_output,
                reference_output,
                precision,
                atol=5e-6 if precision == np.float32 else 1e-3,
            )


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("traditional", [True, False])
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_rope_week2_batch_offset(
    stream: mx.Stream, traditional: bool, precision: np.dtype
):
    BATCH_SIZE = 1
    NUM_HEADS = 8
    HEAD_DIM = 4
    MAX_SEQ_LEN = 20
    SEQ_LEN = 10
    BASE = 10000.0
    with mx.stream(stream):
        for _ in range(100):
            user_layer = RoPE(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=traditional)
            x = np.random.rand(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM).astype(
                precision
            )
            input_pos_user = [slice(i, i + SEQ_LEN) for i in range(BATCH_SIZE)]
            user_layer(mx.array(x), input_pos_user)
