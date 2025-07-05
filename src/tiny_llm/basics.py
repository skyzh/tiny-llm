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
    result = mx.matmul(x, w.T)
    if bias is not None:
        result = result + bias
    return result


def silu(x: mx.array) -> mx.array:
    pass
