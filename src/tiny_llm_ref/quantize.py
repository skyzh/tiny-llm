import mlx.core as mx
from typing import Any
from extensions_ref import tiny_llm_ext_ref


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
            scales=mlx_layer.scales.astype(mx.float16),
            biases=mlx_layer.biases.astype(mx.float16),
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    if bias is not None:
        return (
            quantized_matmul(
                w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
            )
            + bias
        )
    else:
        return quantized_matmul(
            w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
        )


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    *N, D = a.shape
    a = a.reshape(-1, D)
    a = mx.contiguous(a)
    b = mx.contiguous(b)
    return tiny_llm_ext_ref.quantized_matmul(
        scales, biases, group_size, bits, a, b, transpose_b
    ).reshape(*N, -1)
