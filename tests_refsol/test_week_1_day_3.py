import pytest
import mlx.core as mx
import numpy as np
from .tiny_llm_base import *
from .utils import *


def deterministic_array(shape: tuple[int, ...], scale: float, dtype: mx.Dtype):
    values = mx.arange(int(np.prod(shape)), dtype=mx.float32).reshape(shape)
    return (mx.sin(values * 0.37) * scale).astype(dtype)


def qwen_attention_oracle(
    x: mx.array,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    wq: mx.array,
    wk: mx.array,
    wv: mx.array,
    wo: mx.array,
    q_norm: mx.array,
    k_norm: mx.array,
    mask: mx.array | None,
    theta: int,
    rms_norm_eps: float,
) -> mx.array:
    """Independent public-operation oracle for the Day 3 Qwen attention path."""
    B, L, _ = x.shape
    query = mx.matmul(x, wq.T).reshape(B, L, num_heads, head_dim)
    key = mx.matmul(x, wk.T).reshape(B, L, num_kv_heads, head_dim)
    value = mx.matmul(x, wv.T).reshape(B, L, num_kv_heads, head_dim)

    query = mx.fast.rms_norm(query, q_norm, eps=rms_norm_eps)
    key = mx.fast.rms_norm(key, k_norm, eps=rms_norm_eps)
    query = mx.fast.rope(
        query.transpose(0, 2, 1, 3),
        dims=head_dim,
        traditional=False,
        base=theta,
        scale=1.0,
        offset=0,
    )
    key = mx.fast.rope(
        key.transpose(0, 2, 1, 3),
        dims=head_dim,
        traditional=False,
        base=theta,
        scale=1.0,
        offset=0,
    )
    value = value.transpose(0, 2, 1, 3)

    query = query.astype(mx.float32)
    key = key.astype(mx.float32)
    value = value.astype(mx.float32)
    repeats = num_heads // num_kv_heads
    query = query.reshape(B, num_kv_heads, repeats, L, head_dim)
    key = key.reshape(B, num_kv_heads, 1, L, head_dim)
    value = value.reshape(B, num_kv_heads, 1, L, head_dim)
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * (head_dim**-0.5)
    if mask is not None:
        scores = scores + mask.reshape(B, num_kv_heads, repeats, L, L)
    output = mx.matmul(mx.softmax(scores, axis=-1), value)
    output = output.reshape(B, num_heads, L, head_dim).astype(x.dtype)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, num_heads * head_dim)
    return mx.matmul(output, wo.T)


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
        BATCH_3 = 3
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
        elif batch_dimension == 3:
            q_shape = (BATCH_3, BATCH_2, BATCH, H_q, L, D)
            kv_shape = (BATCH_3, BATCH_2, BATCH, H, S, D)
            mask_shape = (BATCH_3, BATCH_2, BATCH, H_q, L, S)
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
    "batch_dimension",
    [0, 1, 2, 3],
    ids=["batch_0", "batch_1", "batch_2", "batch_3"],
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_1_grouped_attention(
    stream: mx.Stream, precision: mx.Dtype, batch_dimension: int, scale: float | None
):
    grouped_attention_helper(stream, precision, batch_dimension, scale, False)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_1_mqa_and_non_divisible_heads(stream: mx.Stream):
    with mx.stream(stream):
        query = deterministic_array((2, 6, 3, 4), 0.5, mx.float32)
        key = deterministic_array((2, 1, 5, 4), 0.4, mx.float32)
        value = deterministic_array((2, 1, 5, 4), 0.3, mx.float32)
        expected = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=4**-0.5
        )
        actual = scaled_dot_product_attention_grouped(query, key, value)
        assert_allclose(actual, expected, precision=mx.float32)

        try:
            scaled_dot_product_attention_grouped(
                query[:, :5],
                mx.broadcast_to(key, (2, 2, 5, 4)),
                mx.broadcast_to(value, (2, 2, 5, 4)),
            )
        except Exception:
            pass
        else:
            raise AssertionError(
                "non-divisible query and key/value heads were accepted"
            )


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
def test_task_2_mask_dtype_and_causal_domain(stream: mx.Stream):
    with mx.stream(stream):
        mask = causal_mask(2, 4, mx.bfloat16)
        assert mask.dtype == mx.bfloat16

        query = mx.ones((2, 5, 4), dtype=mx.float32)
        key = mx.ones((1, 3, 4), dtype=mx.float32)
        value = mx.ones((1, 3, 4), dtype=mx.float32)
        try:
            scaled_dot_product_attention_grouped(query, key, value, mask="causal")
        except Exception:
            pass
        else:
            raise AssertionError("causal attention accepted more queries than keys")


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension",
    [0, 1, 2, 3],
    ids=["batch_0", "batch_1", "batch_2", "batch_3"],
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_2_grouped_attention_causal_mask(
    stream: mx.Stream, precision: mx.Dtype, batch_dimension: int, scale: float | None
):
    grouped_attention_helper(stream, precision, batch_dimension, scale, True)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("mask", [None, "causal"], ids=["no_mask", "causal_mask"])
def test_task_3_qwen3_grouped_query_attention(
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

        from mlx_lm.models import qwen3

        args = qwen3.ModelArgs(
            model_type="qwen3",
            hidden_size=hidden_size,
            num_hidden_layers=2,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=hidden_size // num_heads,
            rms_norm_eps=1e-6,
            vocab_size=1000,
            rope_theta=theta,
            max_position_embeddings=max_seq_len,
            tie_word_embeddings=True,
        )

        mlx_attention = qwen3.Attention(args)
        wq = mlx_attention.q_proj.weight
        wk = mlx_attention.k_proj.weight
        wv = mlx_attention.v_proj.weight
        wo = mlx_attention.o_proj.weight
        q_norm = mlx_attention.q_norm.weight
        k_norm = mlx_attention.k_norm.weight
        mx.random.seed(42)
        x = mx.random.uniform(
            -1.0, 1.0, shape=(batch_size, seq_len, hidden_size), dtype=precision
        )

        user_attention = qwen3_week1.Qwen3MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=hidden_size // num_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=1e-6,
        )

        user_output = user_attention(x, mask=mask)
        mlx_output = mlx_attention(x, mask=mask, cache=None)

        assert_allclose(user_output, mlx_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_3_qwen_public_shape_norm_and_additive_mask(stream: mx.Stream):
    with mx.stream(stream):
        B, L, hidden_size = 2, 4, 12
        num_heads, num_kv_heads, head_dim = 4, 1, 4
        theta = 10000
        eps = 1e-6
        x = deterministic_array((B, L, hidden_size), 0.4, mx.float32)
        wq = deterministic_array((num_heads * head_dim, hidden_size), 0.3, mx.float32)
        wk = deterministic_array(
            (num_kv_heads * head_dim, hidden_size), 0.2, mx.float32
        )
        wv = deterministic_array(
            (num_kv_heads * head_dim, hidden_size), 0.25, mx.float32
        )
        wo = deterministic_array((hidden_size, num_heads * head_dim), 0.2, mx.float32)
        q_norm = mx.array([0.4, 0.9, 1.3, 1.8], dtype=mx.float32)
        k_norm = mx.array([1.7, 1.2, 0.8, 0.3], dtype=mx.float32)
        mask = deterministic_array((B, num_heads, L, L), 0.2, mx.float32)

        layer = qwen3_week1.Qwen3MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            theta=theta,
            rms_norm_eps=eps,
        )
        expected = qwen_attention_oracle(
            x,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            mask=mask,
            theta=theta,
            rms_norm_eps=eps,
        )
        actual = layer(x, mask=mask)
        assert actual.shape == (B, L, hidden_size)
        assert_allclose(actual, expected, precision=mx.float32)


def test_task_3_qwen_bfloat16_attention_arithmetic():
    with mx.stream(mx.gpu):
        B, L, hidden_size = 2, 8, 16
        num_heads, num_kv_heads, head_dim = 4, 2, 6
        theta = 10000
        eps = 1e-6
        dtype = mx.bfloat16
        x = deterministic_array((B, L, hidden_size), 2.0, dtype)
        wq = deterministic_array((num_heads * head_dim, hidden_size), 1.5, dtype)
        wk = deterministic_array((num_kv_heads * head_dim, hidden_size), 1.7, dtype)
        wv = deterministic_array((num_kv_heads * head_dim, hidden_size), 1.3, dtype)
        wo = deterministic_array((hidden_size, num_heads * head_dim), 1.1, dtype)
        q_norm = deterministic_array((head_dim,), 1.4, dtype) + mx.array(1.7, dtype)
        k_norm = deterministic_array((head_dim,), 1.2, dtype) + mx.array(1.5, dtype)

        layer = qwen3_week1.Qwen3MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            theta=theta,
            rms_norm_eps=eps,
        )
        expected = qwen_attention_oracle(
            x,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            mask=None,
            theta=theta,
            rms_norm_eps=eps,
        )
        actual = layer(x)
        assert actual.dtype == dtype
        assert_allclose(actual, expected, precision=dtype, rtol=0.01, atol=0.01)
