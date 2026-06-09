import mlx.core as mx
from .basics import linear
from .quantize import QuantizedWeights, quantized_linear


class Embedding:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        weight: mx.array,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x, :]

    def as_linear(self, x: mx.array) -> mx.array:
        return linear(x, self.weight)


class QuantizedEmbedding:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        weight: QuantizedWeights,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        biases = self.weight.biases[x] if self.weight.biases is not None else None
        return mx.dequantize(
            self.weight.weight[x],
            self.weight.scales[x],
            biases,
            self.weight.group_size,
            self.weight.bits,
        )

    def as_linear(self, x: mx.array) -> mx.array:
        return quantized_linear(x, self.weight)
