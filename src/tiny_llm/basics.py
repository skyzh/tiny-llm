import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    return mx.matmul(x, w.T) + (bias if bias is not None else 0)


def silu(x: mx.array) -> mx.array:
    return x / (1 + mx.exp(-x))
