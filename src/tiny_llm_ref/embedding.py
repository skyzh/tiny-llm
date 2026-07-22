import mlx.core as mx
from extensions_ref import tiny_llm_ext_ref
from .basics import linear
from .quantize import QuantizedWeights, dequantize_weights, quantized_linear


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
        use_custom_kernel: bool = False,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight
        self.use_custom_kernel = use_custom_kernel

    def __call__(self, x: mx.array) -> mx.array:
        if not self.use_custom_kernel or self.weight.biases is None:
            return dequantize_weights(
                self.weight.weight[x],
                self.weight.scales[x],
                None if self.weight.biases is None else self.weight.biases[x],
                self.weight.group_size,
                self.weight.bits,
            )
        return tiny_llm_ext_ref.quantized_embedding(
            x,
            self.weight.scales,
            self.weight.biases,
            self.weight.weight,
            self.weight.group_size,
            self.weight.bits,
        )

    def as_linear(self, x: mx.array) -> mx.array:
        return quantized_linear(x, self.weight)
