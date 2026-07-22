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
        return x @ w.swapaxes(-2, -1) + bias
    return x @ w.swapaxes(-2, -1)

def silu(x: mx.array) -> mx.array:
    sigmod = 1 / (1 + mx.exp(-mx.abs(x)))
    sigmod = mx.where(x < 0, 1 - sigmod, sigmod)
    return x * sigmod
