import mlx.core as mx

from .quantize import QuantizedWeights


def grouped_expert_linear(
    x: mx.array,
    w_experts: QuantizedWeights,
    expert_ids: mx.array,
) -> mx.array:
    pass


def route_topk(
    x: mx.array,
    w_router: QuantizedWeights,
    top_k: int,
    norm_topk_prob: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    pass


class Moe:
    def __init__(
        self,
        w_router: QuantizedWeights,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        num_experts_per_tok: int,
        norm_topk_prob: bool = False,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass
