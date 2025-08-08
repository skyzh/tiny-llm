import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass

    def as_linear(self, x: mx.array) -> mx.array:
        pass
