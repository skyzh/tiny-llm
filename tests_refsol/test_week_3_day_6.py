"""Optional Week 3 Day 6 MoE tests."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_moe import (
    Model as MlxQwen3Moe,
    ModelArgs as MlxQwen3MoeArgs,
    Qwen3MoeSparseMoeBlock as MlxQwen3MoeSparseMoeBlock,
)
from mlx_lm.models.switch_layers import SwitchLinear

from .tiny_llm_base import (
    Moe,
    QuantizedWeights,
    dispatch_model,
    grouped_expert_linear,
    route_topk,
)
from .utils import assert_allclose


def test_task_1_grouped_expert_linear():
    mx.random.seed(1)
    scale = 0.25
    x = mx.random.normal(shape=(2, 3, 128), dtype=mx.bfloat16) * scale
    w_experts = mx.random.normal(shape=(3, 64, 128), dtype=mx.bfloat16) * scale
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
    assert_allclose(out, expected, precision=mx.bfloat16, atol=2e-2)


def test_task_2_router_topk():
    mx.random.seed(2)
    scale = 0.25
    x = mx.random.normal(shape=(2, 2, 128), dtype=mx.bfloat16) * scale
    ref = nn.Linear(128, 4, bias=False)
    ref.weight = mx.random.normal(shape=(4, 128), dtype=mx.bfloat16) * scale
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
    aligned_scores = mx.take_along_axis(expected_probs, expert_ids, axis=-1)
    aligned_normalized_scores = aligned_scores / aligned_scores.sum(
        axis=-1,
        keepdims=True,
    )

    assert router_probs.shape == (2, 2, 4)
    assert expert_ids.shape == (2, 2, 2)
    assert expert_scores.shape == (2, 2, 2)
    assert (
        mx.sort(expert_ids, axis=-1).tolist()
        == mx.sort(
            expected_ids,
            axis=-1,
        ).tolist()
    )
    assert_allclose(router_probs, expected_probs, precision=mx.bfloat16)
    assert_allclose(expert_scores, aligned_scores, precision=mx.bfloat16)
    assert_allclose(
        normalized_scores,
        aligned_normalized_scores,
        precision=mx.bfloat16,
    )


def test_task_3_moe():
    mx.random.seed(3)
    scale = 0.25
    x = mx.random.normal(shape=(2, 3, 128), dtype=mx.bfloat16) * scale
    ref = MlxQwen3MoeSparseMoeBlock(
        SimpleNamespace(
            hidden_size=128,
            moe_intermediate_size=128,
            num_experts=3,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
    )
    ref.gate.weight = mx.random.normal(shape=(3, 128), dtype=mx.bfloat16) * scale
    ref.switch_mlp.gate_proj.weight = (
        mx.random.normal(shape=(3, 128, 128), dtype=mx.bfloat16) * scale
    )
    ref.switch_mlp.up_proj.weight = (
        mx.random.normal(shape=(3, 128, 128), dtype=mx.bfloat16) * scale
    )
    ref.switch_mlp.down_proj.weight = (
        mx.random.normal(shape=(3, 128, 128), dtype=mx.bfloat16) * scale
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
    assert_allclose(out, expected, precision=mx.bfloat16, atol=2e-2)


def test_task_4_model_dispatches_dense_and_sparse_moe_layers():
    mx.random.seed(6)
    args = MlxQwen3MoeArgs(
        model_type="qwen3_moe",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        num_experts=3,
        num_experts_per_tok=2,
        decoder_sparse_step=2,
        mlp_only_layers=[0],
        moe_intermediate_size=128,
        rms_norm_eps=1e-5,
        vocab_size=128,
        num_key_value_heads=1,
        head_dim=64,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        max_position_embeddings=256,
        norm_topk_prob=True,
    )
    mlx_model = MlxQwen3Moe(args)
    nn.quantize(mlx_model, group_size=128, bits=4)

    model = dispatch_model(
        "qwen3-30b-a3b",
        mlx_model,
        week=3,
        page_size=16,
        enable_paged_attention=False,
        use_mlx_quantized_linear=True,
    )
    tokens = mx.array([[1, 5, 7]], dtype=mx.int32)
    actual = model(tokens, 0, model.create_kv_cache())
    expected = mlx_model(tokens)
    actual_probs = mx.softmax(actual.astype(mx.float32), axis=-1)
    expected_probs = mx.softmax(expected.astype(mx.float32), axis=-1)
    mx.eval(actual_probs, expected_probs)

    assert actual.shape == expected.shape == (1, 3, 128)
    assert_allclose(
        actual_probs,
        expected_probs,
        precision=mx.bfloat16,
        rtol=5e-2,
        atol=2e-3,
    )
