import mlx.core as mx

from tiny_llm_ref.attention import scaled_dot_product_attention_grouped
from tiny_llm_ref.week2_kernels import scaled_dot_product_attention

from .utils import assert_allclose


def test_fast_attention_matches_grouped_attention():
    query = mx.random.normal((2, 4, 3, 16)).astype(mx.float32)
    key = mx.random.normal((2, 2, 5, 16)).astype(mx.float32)
    value = mx.random.normal((2, 2, 5, 16)).astype(mx.float32)
    mask = mx.broadcast_to(
        mx.array([0, 0, 0, 0, -mx.inf], dtype=mx.float32), (2, 1, 3, 5)
    )
    scale = 16**-0.5
    result = scaled_dot_product_attention(query, key, value, scale, mask)
    expected = scaled_dot_product_attention_grouped(query, key, value, scale, mask)
    assert result.shape == query.shape
    assert_allclose(result, expected, mx.float32, atol=1e-5, rtol=1e-5)
