"""Week 2 Day 4 SIMD-matrix prefill tests."""

import mlx.core as mx

from .tiny_llm_base import Qwen3ModelWeek2, quantized_matmul, quantized_matmul_vanilla
from .utils import assert_allclose, tiny_qwen3_mlx_model


def test_task_1_simd_matmul_checkpoint_runs_the_week2_engine():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="simd-matmul")
    layer = model.layers_inner[0]

    assert model.use_bounded_kv_capacity
    assert layer.self_attn.wq.use_simdgroup_matmul
    assert not layer.self_attn.use_tiled_prefill_attention

    output = model(
        mx.array([[1, 2, 3]], dtype=mx.int32),
        0,
        model.create_kv_cache(capacity=3),
    )
    assert output.dtype == mx.bfloat16


def test_task_2_simdgroup_matmul_matches_readable_partial_tiles_gpu():
    with mx.stream(mx.gpu):
        inputs = mx.random.normal((10, 256)).astype(mx.bfloat16)
        weight = mx.random.normal((97, 256)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        tiled = quantized_matmul(
            scales,
            biases,
            128,
            4,
            inputs,
            packed,
            transpose_b=True,
            use_simdgroup=True,
        )
        readable = quantized_matmul_vanilla(
            scales, biases, 128, 4, inputs, packed, transpose_b=True
        )
        assert_allclose(tiled, readable, mx.bfloat16, atol=0.25, rtol=1e-2)
