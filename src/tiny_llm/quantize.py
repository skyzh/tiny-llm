import mlx.core as mx
from typing import Any


def dequantize_linear(mx_layer: Any, indices: mx.array | None = None) -> mx.array:
    weight = mx_layer.weight
    scales = mx_layer.scales
    biases = mx_layer.biases
    if indices is not None:
        weight = weight[indices]
        scales = scales[indices]
        biases = biases[indices] if biases is not None else None
    w = mx.dequantize(
        weight,
        scales,
        biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    pass


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    pass
