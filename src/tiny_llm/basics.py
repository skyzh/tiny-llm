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
        return mx.matmul(x, w.T)
    else:
        return mx.matmul(x, w.T) + bias

def silu(x: mx.array) -> mx.array:
    def sigmoid(x: mx.array):
        return 1.0 / (1.0 + mx.exp(-x))
    return x * sigmoid(x)

def logsumexp_norm(x: mx.array):
    c = x.max(axis=-1)
    logsumexp = c + mx.log(mx.sum(mx.exp(x - c), axis=-1))
    return mx.exp(x - logsumexp)