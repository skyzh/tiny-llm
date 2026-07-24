"""Week 2 Day 7 split-K quantized-prefill tests."""

import mlx.core as mx

from .tiny_llm_base import Qwen3ModelWeek2, quantized_matmul
from .utils import assert_allclose, tiny_qwen3_mlx_model


def test_split_k_checkpoint_is_cumulative_and_day_6_stays_unsplit():
    day_6 = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="simd-matmul")
    day_7 = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="split-k")

    assert day_6.layers_inner[0].self_attn.wk.use_simdgroup_matmul
    assert not day_6.layers_inner[0].self_attn.wk.use_split_k_matmul
    assert day_7.layers_inner[0].self_attn.wk.use_simdgroup_matmul
    assert day_7.layers_inner[0].self_attn.wk.use_split_k_matmul


def test_split_k_matches_unsplit_qwen_4b_kv_shape_gpu():
    """The optimized case uses Qwen3-4B's hidden and KV projection sizes."""
    with mx.stream(mx.gpu):
        inputs = mx.random.normal((32, 2560)).astype(mx.bfloat16)
        weight = mx.random.normal((1024, 2560)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        expected = mx.quantized_matmul(
            inputs,
            packed,
            scales,
            biases,
            transpose=True,
            group_size=128,
            bits=4,
        )
        split = quantized_matmul(
            scales,
            biases,
            128,
            4,
            inputs,
            packed,
            transpose_b=True,
            use_simdgroup=True,
            use_split_k=True,
        )
        # Each K partition is stored in BF16 before the FP32 reduction. Keep
        # the tolerance at one output BF16 bin for the extra rounding step.
        assert_allclose(split, expected, mx.bfloat16, atol=1.5, rtol=2e-2)


def test_split_k_handles_partial_output_tiles_gpu():
    with mx.stream(mx.gpu):
        inputs = mx.random.normal((17, 2560)).astype(mx.bfloat16)
        weight = mx.random.normal((1032, 2560)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        expected = mx.quantized_matmul(
            inputs,
            packed,
            scales,
            biases,
            transpose=True,
            group_size=128,
            bits=4,
        )
        split = quantized_matmul(
            scales,
            biases,
            128,
            4,
            inputs,
            packed,
            transpose_b=True,
            use_simdgroup=True,
            use_split_k=True,
        )
        # Each K partition is stored in BF16 before the FP32 reduction. Keep
        # the tolerance at one output BF16 bin for the extra rounding step.
        assert_allclose(split, expected, mx.bfloat16, atol=1.5, rtol=2e-2)


def test_split_k_request_falls_back_for_larger_prefill_gpu():
    with mx.stream(mx.gpu):
        # Four row tiles by 80 output tiles already fill the target grid, so
        # another K partition would add reduction overhead without useful work.
        inputs = mx.random.normal((128, 256)).astype(mx.bfloat16)
        weight = mx.random.normal((2560, 256)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        unsplit = quantized_matmul(
            scales,
            biases,
            128,
            4,
            inputs,
            packed,
            transpose_b=True,
            use_simdgroup=True,
        )
        requested = quantized_matmul(
            scales,
            biases,
            128,
            4,
            inputs,
            packed,
            transpose_b=True,
            use_simdgroup=True,
            use_split_k=True,
        )
        mx.eval(requested, unsplit)
        assert mx.array_equal(requested, unsplit).item()
