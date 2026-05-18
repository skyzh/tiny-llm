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
    if bias is not None:
        return mx.matmul(x, w.T) + bias
    else:
        return mx.matmul(x, w.T)


def silu(x: mx.array) -> mx.array:
    # Avoid exp(large positive) when x is a large negative value.
    sigmoid = 1 / (1 + mx.exp(-mx.abs(x)))
    sigmoid = mx.where(x < 0, 1 - sigmoid, sigmoid)
    return x * sigmoid
