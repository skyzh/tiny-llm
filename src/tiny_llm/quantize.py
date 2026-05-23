import torch
from typing import Any


def dequantize_linear(mx_layer: Any) -> torch.Tensor:
    pass


class QuantizedWeights:
    def __init__(
        self,
        scales: torch.Tensor,
        biases: torch.Tensor,
        group_size: int,
        bits: int,
        weight: torch.Tensor,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        pass


def quantized_matmul(
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    bits: int,
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_b: bool = False,
) -> torch.Tensor:
    pass


def quantized_linear(
    x: torch.Tensor,
    w: QuantizedWeights,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    pass
