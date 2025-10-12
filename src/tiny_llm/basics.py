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
    if bias is None:
        bias = 0
    return mx.matmul(x, w.T) + bias


def silu(x: mx.array) -> mx.array:
    pass
