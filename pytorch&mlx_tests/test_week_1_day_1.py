import pytest
import numpy as np
from ..tiny_llm_base import *
from .utils import *
from .backend import *

@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_softmax(stream, precision):
    with be_stream(stream):
        BATCH_SIZE = 10
        DIM = 10
        for _ in range(100):
            x = be_random_uniform(shape=(BATCH_SIZE, DIM), dtype=precision)
            user_output = softmax(x, axis=-1)
            reference_output = be_softmax(x, axis=-1)
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
def test_task_1_simple_attention(
    stream, precision, batch_dimension: int
):
    """
    Test if `scaled_dot_product_attention_simple` can process Q/K/V correctly.
    We assume Q/K/V are of the same dimensions and test different batch dimensions.
    """
    with be_stream(stream):
        if batch_dimension == 0:
            BATCH_SIZE = ()
        elif batch_dimension == 1:
            BATCH_SIZE = (2, 3)
        elif batch_dimension == 2:
            BATCH_SIZE = (2, 3, 3)
        DIM_L = 4
        DIM_D = 5
        for _ in range(100):
            query = be_random_uniform(
                shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision
            )
            key = be_random_uniform(shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision)
            value = be_random_uniform(
                shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision
            )
            reference_output = be_scaled_dot_product_attention(
                q=be_reshape(query, (1, -1, DIM_L, DIM_D)),
                k=be_reshape(key, (1, -1, DIM_L, DIM_D)),
                v=be_reshape(value, (1, -1, DIM_L, DIM_D)),
                scale=1.0 / (DIM_D**0.5),
            )
            reference_output = be_reshape(reference_output, (*BATCH_SIZE, DIM_L, DIM_D))
            user_output = scaled_dot_product_attention_simple(
                query,
                key,
                value,
            )
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
def test_task_1_simple_attention_scale_mask(
    stream, precision, batch_dimension: int
):
    """
    Test if `scaled_dot_product_attention_simple` can process scale and mask correctly.
    """
    with be_stream(stream):
        if batch_dimension == 0:
            BATCH_SIZE = ()
        elif batch_dimension == 1:
            BATCH_SIZE = (2, 3)
        elif batch_dimension == 2:
            BATCH_SIZE = (2, 3, 3)
        DIM_L = 4
        DIM_D = 5
        for _ in range(100):
            query = be_random_uniform(
                shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision
            )
            key = be_random_uniform(shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision)
            value = be_random_uniform(
                shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision
            )
            mask = be_random_uniform(shape=(*BATCH_SIZE, DIM_L, DIM_L), dtype=precision)
            scale = 0.5
            reference_output = be_scaled_dot_product_attention(
                q=be_reshape(query, (1, -1, DIM_L, DIM_D)),
                k=be_reshape(key, (1, -1, DIM_L, DIM_D)),
                v=be_reshape(value, (1, -1, DIM_L, DIM_D)),
                scale=scale,
                mask=be_reshape(mask, (1, -1, DIM_L, DIM_L)),
            )
            reference_output = be_reshape(reference_output, (*BATCH_SIZE, DIM_L, DIM_D))
            user_output = scaled_dot_product_attention_simple(
                query,
                key,
                value,
                scale=scale,
                mask=mask,
            )
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_linear(stream, precision):
    with be_stream(stream):
        BATCH_SIZE = 10
        DIM_Y = 10
        DIM_X = 12
        for _ in range(100):
            x = be_random_uniform(shape=(BATCH_SIZE, DIM_X), dtype=precision)
            w = be_random_uniform(shape=(DIM_Y, DIM_X), dtype=precision)
            b = be_random_uniform(shape=(DIM_Y,), dtype=precision)
            user_output = linear(x, w, b)
            if BACKEND == "mlx" and precision == be_np_to_dtype(np.float16) and stream == AVAILABLE_STREAMS[0]:
                # unsupported
                break
            reference_output = be_addmm(b, x, w)
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_simple_multi_head_attention(stream, precision):
    """
    Test if `MultiHeadAttention` can process everything correctly. We assume Q/K/V are of the same dimensions.
    """
    with be_stream(stream):
        L = 11
        D = 9
        H = 3
        BATCH_SIZE = 10
        for _ in range(100):
            query = be_random_uniform(shape=(BATCH_SIZE, L, H * D), dtype=precision)
            key = be_random_uniform(shape=(BATCH_SIZE, L, H * D), dtype=precision)
            value = be_random_uniform(shape=(BATCH_SIZE, L, H * D), dtype=precision)
            q_proj_weight = be_random_uniform(shape=(H * D, H * D), dtype=precision)
            k_proj_weight = be_random_uniform(shape=(H * D, H * D), dtype=precision)
            v_proj_weight = be_random_uniform(shape=(H * D, H * D), dtype=precision)
            out_proj_weight = be_random_uniform(shape=(H * D, H * D), dtype=precision)
            mask = be_random_uniform(shape=(L, L), dtype=precision)

            # Use backend MultiHeadAttention as reference
            reference_mha = BackendMultiHeadAttention(H * D, H)
            reference_mha.set_weights(q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight)
            reference_output = reference_mha(query, key, value, mask=mask)

            user_output = SimpleMultiHeadAttention(
                H * D,
                H,
                q_proj_weight,
                k_proj_weight,
                v_proj_weight,
                out_proj_weight,
            )(
                query,
                key,
                value,
                mask=mask,
            )
            assert_allclose(user_output, reference_output, precision=precision)