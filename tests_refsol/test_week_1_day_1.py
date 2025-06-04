import pytest
import mlx.core as mx
import mlx.nn as nn
from .tiny_llm_base import *
from .utils import *


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_softmax(stream: mx.Stream, precision: mx.Dtype):
    with mx.stream(stream):
        BATCH_SIZE = 10
        DIM = 10
        for _ in range(100):
            x = mx.random.uniform(shape=(BATCH_SIZE, DIM), dtype=precision)
            user_output = softmax(x, axis=-1)
            reference_output = mx.softmax(x, axis=-1)
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
def test_task_1_simple_attention(
    stream: mx.Stream, precision: mx.Dtype, batch_dimension: int
):
    """
    Test if `scaled_dot_product_attention_simple` can process Q/K/V correctly.
    We assume Q/K/V are of the same dimensions and test different batch dimensions.
    """
    with mx.stream(stream):
        if batch_dimension == 0:
            BATCH_SIZE = ()
        elif batch_dimension == 1:
            BATCH_SIZE = (2, 3)
        elif batch_dimension == 2:
            BATCH_SIZE = (2, 3, 3)
        DIM_L = 4
        DIM_D = 5
        for _ in range(100):
            query = mx.random.uniform(shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision)
            key = mx.random.uniform(shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision)
            value = mx.random.uniform(shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision)
            reference_output = mx.fast.scaled_dot_product_attention(
                q=query.reshape(1, -1, DIM_L, DIM_D),
                k=key.reshape(1, -1, DIM_L, DIM_D),
                v=value.reshape(1, -1, DIM_L, DIM_D),
                scale=1.0 / (DIM_D ** 0.5),
            ).reshape(*BATCH_SIZE, DIM_L, DIM_D)
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
    stream: mx.Stream, precision: mx.Dtype, batch_dimension: int
):
    """
    Test if `scaled_dot_product_attention_simple` can process scale and mask correctly.
    """
    with mx.stream(stream):
        if batch_dimension == 0:
            BATCH_SIZE = ()
        elif batch_dimension == 1:
            BATCH_SIZE = (2, 3)
        elif batch_dimension == 2:
            BATCH_SIZE = (2, 3, 3)
        DIM_L = 4
        DIM_D = 5
        for _ in range(100):
            query = mx.random.uniform(shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision)
            key = mx.random.uniform(shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision)
            value = mx.random.uniform(shape=(*BATCH_SIZE, DIM_L, DIM_D), dtype=precision)
            mask = mx.random.uniform(shape=(*BATCH_SIZE, DIM_L, DIM_L), dtype=precision)
            scale = 0.5
            reference_output = mx.fast.scaled_dot_product_attention(
                q=query.reshape(1, -1, DIM_L, DIM_D),
                k=key.reshape(1, -1, DIM_L, DIM_D),
                v=value.reshape(1, -1, DIM_L, DIM_D),
                scale=scale,
                mask=mask.reshape(1, -1, DIM_L, DIM_L),
            ).reshape(*BATCH_SIZE, DIM_L, DIM_D)
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
def test_task_2_linear(stream: mx.Stream, precision: mx.Dtype):
    with mx.stream(stream):
        BATCH_SIZE = 10
        DIM_Y = 10
        DIM_X = 12
        for _ in range(100):
            x = mx.random.uniform(shape=(BATCH_SIZE, DIM_X), dtype=precision)
            w = mx.random.uniform(shape=(DIM_Y, DIM_X), dtype=precision)
            b = mx.random.uniform(shape=(DIM_Y,), dtype=precision)
            user_output = linear(x, w, b)
            if precision == mx.float16 and stream == mx.cpu:
                # unsupported
                break
            reference_output = mx.addmm(b, x, w.T)
            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_simple_multi_head_attention(stream: mx.Stream, precision: mx.Dtype):
    """
    Test if `MultiHeadAttention` can process everything correctly. We assume Q/K/V are of the same dimensions.
    """
    with mx.stream(stream):
        L = 11
        D = 9
        H = 3
        BATCH_SIZE = 10
        for _ in range(100):
            query = mx.random.uniform(shape=(BATCH_SIZE, L, H * D), dtype=precision)
            key = mx.random.uniform(shape=(BATCH_SIZE, L, H * D), dtype=precision)
            value = mx.random.uniform(shape=(BATCH_SIZE, L, H * D), dtype=precision)
            q_proj_weight = mx.random.uniform(shape=(H * D, H * D), dtype=precision)
            k_proj_weight = mx.random.uniform(shape=(H * D, H * D), dtype=precision)
            v_proj_weight = mx.random.uniform(shape=(H * D, H * D), dtype=precision)
            out_proj_weight = mx.random.uniform(shape=(H * D, H * D), dtype=precision)
            mask = mx.random.uniform(shape=(L, L), dtype=precision)
            
            # Use MLX built-in MultiHeadAttention as reference
            reference_mha = nn.MultiHeadAttention(H * D, H)
            
            # Set the weights manually to match our test case
            reference_mha.query_proj.weight = q_proj_weight
            reference_mha.key_proj.weight = k_proj_weight
            reference_mha.value_proj.weight = v_proj_weight
            reference_mha.out_proj.weight = out_proj_weight
            
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
