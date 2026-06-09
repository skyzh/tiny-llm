from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_moe import (
    Qwen3MoeSparseMoeBlock as MlxQwen3MoeSparseMoeBlock,
)
from mlx_lm.models.switch_layers import SwitchLinear

from .tiny_llm_base import (
    Moe,
    QuantizedWeights,
    grouped_expert_linear,
    route_topk,
)
from .utils import assert_allclose


def test_task_1_grouped_expert_linear():
    mx.random.seed(1)
    scale = 0.25
    x = mx.random.normal(shape=(2, 3, 128), dtype=mx.float16) * scale
    w_experts = mx.random.normal(shape=(3, 64, 128), dtype=mx.float16) * scale
    expert_ids = mx.array(
        [
            [2, 0, 1],
            [1, 2, 0],
        ],
        dtype=mx.uint32,
    )

    ref = SwitchLinear(
        input_dims=w_experts.shape[-1],
        output_dims=w_experts.shape[-2],
        num_experts=w_experts.shape[0],
        bias=False,
    )
    ref.weight = w_experts
    ref = ref.to_quantized(group_size=128, bits=4)

    out = grouped_expert_linear(
        x,
        QuantizedWeights.from_mlx_layer(ref),
        expert_ids,
    )
    expected = ref(mx.expand_dims(x, -2), expert_ids).squeeze(-2)

    assert out.shape == (2, 3, 64)
    assert_allclose(out, expected, precision=mx.float16)


def test_task_2_router_topk():
    mx.random.seed(2)
    scale = 0.25
    x = mx.random.normal(shape=(2, 2, 128), dtype=mx.float16) * scale
    ref = nn.Linear(128, 4, bias=False)
    ref.weight = mx.random.normal(shape=(4, 128), dtype=mx.float16) * scale
    ref = ref.to_quantized(group_size=128, bits=4)

    router_probs, expert_ids, expert_scores = route_topk(
        x,
        QuantizedWeights.from_mlx_layer(ref),
        top_k=2,
    )
    _, _, normalized_scores = route_topk(
        x,
        QuantizedWeights.from_mlx_layer(ref),
        top_k=2,
        norm_topk_prob=True,
    )

    expected_probs = mx.softmax(ref(x), axis=-1, precise=True)
    expected_ids = mx.argpartition(-expected_probs, kth=1, axis=-1)[..., :2]
    expected_scores = mx.take_along_axis(expected_probs, expected_ids, axis=-1)
    expected_normalized_scores = expected_scores / expected_scores.sum(
        axis=-1,
        keepdims=True,
    )

    assert router_probs.shape == (2, 2, 4)
    assert expert_ids.shape == (2, 2, 2)
    assert expert_scores.shape == (2, 2, 2)
    assert expert_ids.tolist() == expected_ids.tolist()
    assert_allclose(router_probs, expected_probs, precision=mx.float16)
    assert_allclose(expert_scores, expected_scores, precision=mx.float16)
    assert_allclose(
        normalized_scores,
        expected_normalized_scores,
        precision=mx.float16,
    )


def test_task_3_moe():
    mx.random.seed(3)
    scale = 0.25
    x = mx.random.normal(shape=(2, 3, 128), dtype=mx.float16) * scale
    ref = MlxQwen3MoeSparseMoeBlock(
        SimpleNamespace(
            hidden_size=128,
            moe_intermediate_size=128,
            num_experts=3,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
    )
    ref.gate.weight = mx.random.normal(shape=(3, 128), dtype=mx.float16) * scale
    ref.switch_mlp.gate_proj.weight = (
        mx.random.normal(shape=(3, 128, 128), dtype=mx.float16) * scale
    )
    ref.switch_mlp.up_proj.weight = (
        mx.random.normal(shape=(3, 128, 128), dtype=mx.float16) * scale
    )
    ref.switch_mlp.down_proj.weight = (
        mx.random.normal(shape=(3, 128, 128), dtype=mx.float16) * scale
    )
    nn.quantize(ref, group_size=128, bits=4)

    moe = Moe(
        w_router=QuantizedWeights.from_mlx_layer(ref.gate),
        w_gate=QuantizedWeights.from_mlx_layer(ref.switch_mlp.gate_proj),
        w_up=QuantizedWeights.from_mlx_layer(ref.switch_mlp.up_proj),
        w_down=QuantizedWeights.from_mlx_layer(ref.switch_mlp.down_proj),
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )

    out = moe(x)
    expected = ref(x)

    assert out.shape == x.shape
    assert_allclose(out, expected, precision=mx.float16)
