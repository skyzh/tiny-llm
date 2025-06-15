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
    output = mx.matmul(x, w.swapaxes(-2, -1))
    if bias is not None:
        output = output + bias
    return output


def silu(x: mx.array) -> mx.array:
    pass
