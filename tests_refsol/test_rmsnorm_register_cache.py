"""Public RMSNorm behavior at and beyond the register-cache boundary."""

import mlx.core as mx
import pytest

from tiny_llm_ref.layer_norm import RMSNorm
from tiny_llm_ref.week2_kernels import FastRMSNorm
from .utils import assert_allclose


@pytest.mark.parametrize("dtype", (mx.float32, mx.float16, mx.bfloat16))
@pytest.mark.parametrize("dim", (7, 16, 128, 2560, 4095, 4096))
def test_rmsnorm_matches_readable_at_register_cache_boundaries(dtype, dim):
    x = mx.sin(mx.arange(dim * 2, dtype=mx.float32) * 0.013).reshape(2, dim)
    weight = mx.cos(mx.arange(dim, dtype=mx.float32) * 0.019)
    x = x.astype(dtype)
    weight = weight.astype(dtype)
    fast = FastRMSNorm(dim, weight, eps=1e-5)

    actual = fast(x)
    expected = RMSNorm(dim, weight, eps=1e-5)(x)

    tolerance = 1e-5 if dtype == mx.float32 else 2e-2
    assert_allclose(actual, expected, dtype, atol=tolerance, rtol=tolerance)


def test_rmsnorm_matches_readable_above_register_cache_limit():
    dim = 4097
    x = mx.sin(mx.arange(dim, dtype=mx.float32) * 0.013).reshape(1, dim)
    weight = mx.ones((dim,), dtype=mx.float32)
    fast = FastRMSNorm(dim, weight, eps=1e-5)

    actual = fast(x)
    expected = RMSNorm(dim, weight, eps=1e-5)(x)

    assert_allclose(actual, expected, mx.float32, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("shape", "weight_shape"),
    (((2, 8), (7,)), ((2, 8), (8, 1))),
)
def test_rmsnorm_rejects_incompatible_weights(shape, weight_shape):
    with pytest.raises(RuntimeError, match="weight must match"):
        result = FastRMSNorm(
            shape[-1],
            mx.ones(weight_shape, dtype=mx.bfloat16),
        )(mx.ones(shape, dtype=mx.bfloat16))
        mx.eval(result)


def test_rmsnorm_rejects_non_float_and_empty_final_dimensions():
    with pytest.raises(RuntimeError, match="expected float32, float16, or bfloat16"):
        result = FastRMSNorm(8, mx.ones((8,), dtype=mx.int32))(
            mx.ones((2, 8), dtype=mx.int32)
        )
        mx.eval(result)

    with pytest.raises(RuntimeError, match="weight must match"):
        result = FastRMSNorm(0, mx.ones((0,), dtype=mx.float32))(
            mx.ones((2, 0), dtype=mx.float32)
        )
        mx.eval(result)
