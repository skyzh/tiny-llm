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
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
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
    return mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )


def dequantize_weights(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    group_size: int,
    bits: int,
) -> mx.array:
    if bits <= 0 or 32 % bits != 0:
        raise ValueError("bits must divide a 32-bit packed weight")
    values_per_word = 32 // bits
    shifts = mx.arange(0, 32, bits, dtype=mx.uint32)
    values = (weight[..., None] >> shifts) & ((1 << bits) - 1)
    values = values.reshape(*weight.shape[:-1], weight.shape[-1] * values_per_word)
    values = values.astype(mx.float32)
    expanded_scales = mx.repeat(scales, group_size, axis=-1).astype(mx.float32)
    if biases is None:
        return (values * expanded_scales).astype(scales.dtype)
    expanded_biases = mx.repeat(biases, group_size, axis=-1).astype(mx.float32)
    return (values * expanded_scales + expanded_biases).astype(scales.dtype)


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
    result = tiny_llm_ext_ref.quantized_matmul(
        mx.contiguous(scales),
        mx.contiguous(biases),
        group_size,
        bits,
        mx.contiguous(a),
        mx.contiguous(b),
        transpose_b,
    )
    return result.reshape(*N, -1)


def quantized_matvec_custom(
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
    if a.shape[0] > 8:
        raise ValueError("quantized_matvec_custom supports at most 8 input rows")
    result = tiny_llm_ext_ref.quantized_matmul(
        mx.contiguous(scales),
        mx.contiguous(biases),
        group_size,
        bits,
        mx.contiguous(a),
        mx.contiguous(b),
        transpose_b,
    )
    return result.reshape(*N, -1)
