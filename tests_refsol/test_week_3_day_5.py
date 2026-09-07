"""Week 3 Day 5 paged-FlashAttention public-behavior tests."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from .tiny_llm_base import (
    Qwen3ModelWeek3,
    paged_attention,
    scaled_dot_product_attention_grouped,
)
from .utils import assert_allclose


def _assert_paged_case(
    *,
    query_length: int,
    context_length: int,
    page_size: int,
    physical_orders: list[list[int]],
    head_dim: int = 128,
    scale: float | None = None,
    mask: str | None = "causal",
) -> None:
    """Compare public paged attention with its readable dense equation."""
    mx.random.seed(query_length + context_length + head_dim)
    batch_size = len(physical_orders)
    num_kv_heads = 2
    num_heads = 4

    query = mx.random.normal((batch_size, num_heads, query_length, head_dim)).astype(
        mx.bfloat16
    )
    dense_key = mx.random.normal(
        (batch_size, num_kv_heads, context_length, head_dim)
    ).astype(mx.bfloat16)
    dense_value = mx.random.normal(
        (batch_size, num_kv_heads, context_length, head_dim)
    ).astype(mx.bfloat16)

    num_physical_pages = max(page for order in physical_orders for page in order) + 2
    # Poison unused physical pages and tail slots. Only logical live positions
    # named by block_table may affect the result.
    key_pages = (
        mx.ones(
            (num_physical_pages, num_kv_heads, page_size, head_dim),
            dtype=mx.bfloat16,
        )
        * 29
    )
    value_pages = (
        mx.ones(
            (num_physical_pages, num_kv_heads, page_size, head_dim),
            dtype=mx.bfloat16,
        )
        * -23
    )
    for batch, physical_order in enumerate(physical_orders):
        for logical_page, physical_page in enumerate(physical_order):
            start = logical_page * page_size
            stop = min(start + page_size, context_length)
            width = stop - start
            key_pages[physical_page, :, :width, :] = dense_key[batch, :, start:stop, :]
            value_pages[physical_page, :, :width, :] = dense_value[
                batch, :, start:stop, :
            ]

    expected = scaled_dot_product_attention_grouped(
        query,
        dense_key,
        dense_value,
        scale=scale,
        mask=mask,
    )
    actual = paged_attention(
        query,
        key_pages,
        value_pages,
        mx.array(physical_orders, dtype=mx.int32),
        mx.array([context_length] * batch_size, dtype=mx.int32),
        page_size,
        scale=scale,
        mask=mask,
    )
    mx.eval(expected, actual)

    assert actual.shape == query.shape
    assert actual.dtype == mx.bfloat16
    assert_allclose(actual, expected, mx.bfloat16, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("query_length", "context_length", "page_size", "physical_orders"),
    [
        (9, 9, 16, [[2]]),
        (65, 97, 16, [[5, 1, 6, 0, 4, 2, 3]]),
    ],
    ids=["one-page", "partial-query-and-tail"],
)
def test_long_prefill_matches_dense_attention_across_paged_boundaries(
    query_length: int,
    context_length: int,
    page_size: int,
    physical_orders: list[list[int]],
) -> None:
    _assert_paged_case(
        query_length=query_length,
        context_length=context_length,
        page_size=page_size,
        physical_orders=physical_orders,
    )


def test_batched_gqa_long_prefill_honors_explicit_scale() -> None:
    _assert_paged_case(
        query_length=17,
        context_length=41,
        page_size=16,
        physical_orders=[[4, 0, 5], [2, 6, 1]],
        scale=0.125,
        mask=None,
    )


@pytest.mark.parametrize(
    ("query_length", "context_length", "head_dim", "mask"),
    [(1, 17, 128, "causal"), (9, 17, 32, None)],
    ids=["decode", "generic-bfloat16-prefill"],
)
def test_day5_preserves_decode_and_generic_bfloat16_fallback(
    query_length: int,
    context_length: int,
    head_dim: int,
    mask: str | None,
) -> None:
    _assert_paged_case(
        query_length=query_length,
        context_length=context_length,
        page_size=16,
        physical_orders=[[1, 0]],
        head_dim=head_dim,
        scale=0.25,
        mask=mask,
    )


def _quantized_layer(out_dim: int, in_dim: int) -> SimpleNamespace:
    weight = (mx.random.normal((out_dim, in_dim)) * 0.05).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
    return SimpleNamespace(
        weight=packed,
        scales=scales,
        biases=biases,
        group_size=128,
        bits=4,
    )


def _fake_qwen3_mlx_model() -> SimpleNamespace:
    hidden_size = 128
    head_dim = 128
    intermediate_size = 256
    args = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=hidden_size,
        vocab_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
        rope_theta=10000,
        tie_word_embeddings=True,
    )
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            q_proj=_quantized_layer(hidden_size, hidden_size),
            k_proj=_quantized_layer(hidden_size, hidden_size),
            v_proj=_quantized_layer(hidden_size, hidden_size),
            o_proj=_quantized_layer(hidden_size, hidden_size),
            q_norm=SimpleNamespace(weight=mx.ones((head_dim,), dtype=mx.bfloat16)),
            k_norm=SimpleNamespace(weight=mx.ones((head_dim,), dtype=mx.bfloat16)),
        ),
        mlp=SimpleNamespace(
            gate_proj=_quantized_layer(intermediate_size, hidden_size),
            up_proj=_quantized_layer(intermediate_size, hidden_size),
            down_proj=_quantized_layer(hidden_size, intermediate_size),
        ),
        input_layernorm=SimpleNamespace(
            weight=mx.ones((hidden_size,), dtype=mx.bfloat16)
        ),
        post_attention_layernorm=SimpleNamespace(
            weight=mx.ones((hidden_size,), dtype=mx.bfloat16)
        ),
    )
    return SimpleNamespace(
        args=args,
        model=SimpleNamespace(
            embed_tokens=_quantized_layer(args.vocab_size, hidden_size),
            layers=[layer],
            norm=SimpleNamespace(weight=mx.ones((hidden_size,), dtype=mx.bfloat16)),
        ),
    )


def test_week3_attention_module_runs_long_prefill_through_paged_storage() -> None:
    """The Week 3 model keeps the Day 4/5 paged-attention boundary."""
    mx.random.seed(7)
    mlx_model = _fake_qwen3_mlx_model()
    paged_model = Qwen3ModelWeek3(
        mlx_model,
        page_size=16,
        enable_paged_attention=True,
        use_mlx_quantized_linear=False,
    )
    dense_model = Qwen3ModelWeek3(
        mlx_model,
        page_size=16,
        enable_paged_attention=False,
        use_mlx_quantized_linear=False,
    )
    inputs = mx.array([[1, 5, 7, 3, 9, 11, 13, 15, 17]], dtype=mx.int32)

    paged_output = paged_model(inputs, 0, paged_model.create_kv_cache())
    dense_output = dense_model(inputs, 0, dense_model.create_kv_cache())
    mx.eval(paged_output, dense_output)

    assert paged_output.shape == dense_output.shape == (1, 9, 128)
    assert paged_output.dtype == dense_output.dtype == mx.bfloat16
    assert_allclose(
        paged_output,
        dense_output,
        mx.bfloat16,
        rtol=3e-2,
        atol=3e-2,
    )
