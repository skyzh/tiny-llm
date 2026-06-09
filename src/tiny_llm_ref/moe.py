import mlx.core as mx

from .basics import silu
from .quantize import QuantizedWeights, quantized_linear


def grouped_expert_linear(
    x: mx.array,
    w_experts: QuantizedWeights,
    expert_ids: mx.array,
) -> mx.array:
    *leading_dims, D = x.shape
    flat_x = x.reshape(-1, D)
    flat_expert_ids = expert_ids.reshape(-1)
    sort_idx = mx.argsort(flat_expert_ids)
    inv_sort_idx = mx.argsort(sort_idx)

    grouped_x = flat_x[sort_idx]
    grouped_expert_ids = flat_expert_ids[sort_idx]
    out = mx.gather_qmm(
        mx.expand_dims(grouped_x, -2),
        w_experts.weight,
        w_experts.scales,
        w_experts.biases,
        lhs_indices=mx.arange(grouped_x.shape[0]),
        rhs_indices=grouped_expert_ids,
        transpose=True,
        group_size=w_experts.group_size,
        bits=w_experts.bits,
        sorted_indices=True,
    ).squeeze(-2)
    out_dim = w_experts.weight.shape[-2]
    return out[inv_sort_idx].reshape(*leading_dims, out_dim)


def route_topk(
    x: mx.array,
    w_router: QuantizedWeights,
    top_k: int,
    norm_topk_prob: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    router_logits = quantized_linear(x, w_router)
    router_probs = mx.softmax(router_logits, axis=-1, precise=True)
    expert_ids = mx.argpartition(-router_probs, kth=top_k - 1, axis=-1)[..., :top_k]
    expert_scores = mx.take_along_axis(router_probs, expert_ids, axis=-1)
    if norm_topk_prob:
        expert_scores = expert_scores / mx.sum(expert_scores, axis=-1, keepdims=True)
    return router_probs, expert_ids, expert_scores


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
        self.w_router = w_router
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        _, expert_ids, expert_scores = route_topk(
            x,
            self.w_router,
            top_k=self.num_experts_per_tok,
            norm_topk_prob=self.norm_topk_prob,
        )
        expanded_x = mx.broadcast_to(
            mx.expand_dims(x, -2),
            (B, L, self.num_experts_per_tok, D),
        ).reshape(-1, D)
        flat_expert_ids = expert_ids.reshape(-1)

        gate = grouped_expert_linear(expanded_x, self.w_gate, flat_expert_ids)
        up = grouped_expert_linear(expanded_x, self.w_up, flat_expert_ids)
        expert_output = grouped_expert_linear(
            silu(gate) * up,
            self.w_down,
            flat_expert_ids,
        ).reshape(B, L, self.num_experts_per_tok, D)
        return mx.sum(expert_output * mx.expand_dims(expert_scores, -1), axis=-2)
