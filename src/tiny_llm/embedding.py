import mlx.core as mx
from .basics import linear


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        assert weight.shape[0] == vocab_size
        assert weight.shape[1] == embedding_dim

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x, :]

    def as_linear(self, x: mx.array) -> mx.array:
        return linear(x, self.weight)
