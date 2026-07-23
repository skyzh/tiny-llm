"""Week 3 Day 5 paged-FlashAttention tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import (
    TinyKvPagedCache,
    TinyKvPagedPool,
    paged_attention,
    scaled_dot_product_attention_grouped,
)
from .utils import assert_allclose


def _random_chunk(
    length: int, num_heads: int = 2, head_dim: int = 128
) -> tuple[mx.array, mx.array]:
    key = mx.random.normal((1, num_heads, length, head_dim)).astype(mx.bfloat16)
    value = mx.random.normal((1, num_heads, length, head_dim)).astype(mx.bfloat16)
    return key, value


@pytest.mark.parametrize("query_length", [9, 65])
def test_paged_flash_attention_crosses_noncontiguous_pages(query_length: int):
    page_size = 32
    pool = TinyKvPagedPool(page_size=page_size)
    cache = TinyKvPagedCache(pool=pool)
    blocker = TinyKvPagedCache(pool=pool)

    cache.update_and_fetch(*_random_chunk(64))
    blocker.update_and_fetch(*_random_chunk(page_size))
    next_key, next_value = _random_chunk(query_length)
    metadata = cache.update_and_fetch_paged(
        next_key,
        next_value,
        mask="causal",
    )
    query = mx.random.normal((1, 4, query_length, 128)).astype(mx.bfloat16)

    dense_key, dense_value = cache.gather_dense()
    expected = scaled_dot_product_attention_grouped(
        query,
        dense_key,
        dense_value,
        mask="causal",
    )
    actual = paged_attention(
        query,
        metadata.key_pages,
        metadata.value_pages,
        metadata.block_table,
        metadata.context_lens,
        metadata.page_size,
        mask=metadata.mask,
    )
    mx.eval(expected, actual)

    assert cache.page_ids[:2] == [0, 1]
    assert cache.page_ids[2] == 3
    assert actual.dtype == mx.bfloat16
    assert_allclose(actual, expected, mx.bfloat16, rtol=2e-2, atol=2e-2)
