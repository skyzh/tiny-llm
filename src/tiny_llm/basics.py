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
    # x: 1, E
    # w: D, E
    # b: 1, D
    if bias is None:
        bias = 0
    # (1, E) * (E, D) -> (1, D)
    return mx.matmul(x, w.T) + bias


def silu(x: mx.array) -> mx.array:
    # silu(x) = x * sigmoid(x) = x/(1 + e^-x)
    return x / (1 + mx.exp(-x))
