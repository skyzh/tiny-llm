import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


def grouped_attention_helper(
    stream: mx.Stream,
    precision: mx.Dtype,
    batch_dimension: int,
    scale: float | None,
    is_causal_mask: bool,
):
    with mx.stream(stream):
        H_q = 18
        H = 6
        L = 3
        D = 5
        S = 7
        BATCH = 10
        BATCH_2 = 2
        if batch_dimension == 0:
            q_shape = (H_q, L, D)
            kv_shape = (H, S, D)
            mask_shape = (H_q, L, S)
        elif batch_dimension == 1:
            q_shape = (BATCH, H_q, L, D)
            kv_shape = (BATCH, H, S, D)
            mask_shape = (BATCH, H_q, L, S)
        elif batch_dimension == 2:
            q_shape = (BATCH_2, BATCH, H_q, L, D)
            kv_shape = (BATCH_2, BATCH, H, S, D)
            mask_shape = (BATCH_2, BATCH, H_q, L, S)
        for _ in range(100):
            query = mx.random.uniform(shape=q_shape, dtype=precision)
            key = mx.random.uniform(shape=kv_shape, dtype=precision)
            value = mx.random.uniform(shape=kv_shape, dtype=precision)
            mask = mx.random.uniform(shape=mask_shape, dtype=precision)

            reference_output = mx.fast.scaled_dot_product_attention(
                q=query.reshape(-1, H_q, L, D),
                k=key.reshape(-1, H, S, D),
                v=value.reshape(-1, H, S, D),
                scale=scale if scale is not None else (1.0 / (D**0.5)),
                mask=mask.reshape(-1, H_q, L, S) if not is_causal_mask else "causal",
            )
            # Reshape reference output back to original shape
            reference_output = reference_output.reshape(query.shape)
            user_output = scaled_dot_product_attention_grouped(
                query,
                key,
                value,
                scale=scale,
                mask=mask if not is_causal_mask else "causal",
            )

            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_1_grouped_attention(
    stream: mx.Stream, precision: mx.Dtype, batch_dimension: int, scale: float | None
):
    grouped_attention_helper(stream, precision, batch_dimension, scale, False)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_2_mask_only_same_dim(
    stream: mx.Stream,
):
    with mx.stream(stream):
        L = 3
        S = 3
        user_output = causal_mask(
            L,
            S,
            mx.float32,
        )
        assert_allclose(
            user_output,
            mx.array(
                [
                    [0, -mx.inf, -mx.inf],
                    [0, 0, -mx.inf],
                    [0, 0, 0],
                ]
            ),
            precision=mx.float32,
        )


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_2_mask_only_different_dim(
    stream: mx.Stream,
):
    with mx.stream(stream):
        L = 3
        S = 5
        user_output = causal_mask(
            L,
            S,
            mx.float32,
        )
        assert_allclose(
            user_output,
            mx.array(
                [
                    [0, 0, 0, -mx.inf, -mx.inf],
                    [0, 0, 0, 0, -mx.inf],
                    [0, 0, 0, 0, 0],
                ]
            ),
            precision=mx.float32,
        )


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_2_grouped_attention_causal_mask(
    stream: mx.Stream, precision: mx.Dtype, batch_dimension: int, scale: float | None
):
    grouped_attention_helper(stream, precision, batch_dimension, scale, True)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("mask", [None, "causal"], ids=["no_mask", "causal_mask"])
def test_task_3_qwen2_grouped_query_attention(
    stream: mx.Stream, precision: mx.Dtype, mask: str | None
):
    with mx.stream(stream):
        batch_size = 1
        seq_len = 4
        hidden_size = 32
        num_heads = 4
        num_kv_heads = 2
        max_seq_len = 64
        theta = 10000

        from mlx_lm.models import qwen2

        args = qwen2.ModelArgs(
            model_type="qwen2",
            hidden_size=hidden_size,
            num_hidden_layers=2,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            rms_norm_eps=1e-6,
            vocab_size=1000,
            rope_theta=theta,
            rope_traditional=False,
            max_position_embeddings=max_seq_len,
        )

        mlx_attention = qwen2.Attention(args)
        wq = mlx_attention.q_proj.weight
        wk = mlx_attention.k_proj.weight
        wv = mlx_attention.v_proj.weight
        wo = mlx_attention.o_proj.weight
        bq = mlx_attention.q_proj.bias
        bk = mlx_attention.k_proj.bias
        bv = mlx_attention.v_proj.bias
        mx.random.seed(42)
        x = mx.random.uniform(
            -1.0, 1.0, shape=(batch_size, seq_len, hidden_size), dtype=precision
        )

        user_attention = qwen2_week1.Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        user_output = user_attention(x, mask=mask)
        mlx_output = mlx_attention(x, mask=mask, cache=None)

        assert_allclose(user_output, mlx_output, precision=precision)
