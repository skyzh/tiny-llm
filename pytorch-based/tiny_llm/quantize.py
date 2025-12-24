import torch
from typing import Any


def dequantize_linear(torch_layer: Any) -> torch.Tensor:
    q_weight = torch_layer.weight
    scales = torch_layer.scales
    zero_points = torch_layer.biases
    group_size = torch_layer.group_size
    bits = torch_layer.bits

    num_elements = q_weight.numel()
    num_groups = num_elements // group_size
    q_weight = q_weight.view(num_groups, group_size)

    scales = scales.view(-1, 1)
    zero_points = zero_points.view(-1, 1)

    dequantized = scales * (q_weight.float() - zero_points)

    return dequantized.view(-1)
