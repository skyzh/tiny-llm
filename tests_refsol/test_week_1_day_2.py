import pytest
import mlx.core as mx
from .tiny_llm_base import *
import numpy as np
from .utils import *


def rope_helper(
    stream: mx.Stream,
    traditional: bool,
    precision: mx.Dtype,
    with_offset: bool,
):
    BATCH_SIZE = 1
    NUM_HEADS = 8
    HEAD_DIM = 4
    MAX_SEQ_LEN = 20
    SEQ_LEN = 10
    BASE = 10000
    with mx.stream(stream):
        for _ in range(100):
            user_layer = RoPE(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=traditional)
            x = mx.random.uniform(
                shape=(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=precision
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
                x.transpose(0, 2, 1, 3),
                dims=HEAD_DIM,
                traditional=traditional,
                base=BASE,
                scale=1.0,
                offset=input_pos_mx or 0,
            ).transpose(0, 2, 1, 3)
            user_output = user_layer(x, input_pos_user)
            assert_allclose(
                user_output,
                reference_output,
                precision,
                atol=5e-6 if precision == mx.float32 else 1e-3,
            )


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize(
    "with_offset", [True, False], ids=["with_offset", "without_offset"]
)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rope_mlx_traditional(
    stream: mx.Stream, with_offset: bool, precision: mx.Dtype
):
    rope_helper(stream, True, precision, with_offset)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize(
    "with_offset", [True, False], ids=["with_offset", "without_offset"]
)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_rope_mlx_non_traditional(
    stream: mx.Stream, with_offset: bool, precision: mx.Dtype
):
    rope_helper(stream, False, precision, with_offset)
