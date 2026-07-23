"""Week 2 Day 3 quantized-matvec tests."""

import importlib
import inspect

import mlx.core as mx

from .tiny_llm_base import (
    Qwen3ModelWeek2,
    QuantizedEmbedding,
    QuantizedWeights,
    RMSNorm,
    RoPE,
    quantized_matmul,
    quantized_matmul_vanilla,
    quantized_matvec_custom,
)
from .utils import assert_allclose, tiny_qwen3_mlx_model

embedding_module = importlib.import_module(QuantizedEmbedding.__module__)
quantize_module = importlib.import_module(quantized_matmul.__module__)


def test_task_1_quantized_embedding_dequantizes_selected_rows():
    weight = mx.random.normal((7, 256)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
    embedding = QuantizedEmbedding(
        7, 256, QuantizedWeights(scales, biases, 128, 4, packed)
    )
    indices = mx.array([[1, 4]])

    result = embedding(indices)
    expected = mx.dequantize(
        packed[indices], scales[indices], biases[indices], group_size=128, bits=4
    )
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


def test_task_1_quantized_embedding_accepts_sampled_uint32_tokens():
    weight = mx.random.normal((7, 256)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
    embedding = QuantizedEmbedding(
        7, 256, QuantizedWeights(scales, biases, 128, 4, packed)
    )
    indices = mx.array([[1, 4]], dtype=mx.uint32)

    result = embedding(indices)
    expected = mx.dequantize(
        packed[indices], scales[indices], biases[indices], group_size=128, bits=4
    )
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


def test_week2_quantization_path_uses_course_owned_operators():
    source = (
        inspect.getsource(quantize_module.quantized_matmul)
        + inspect.getsource(quantize_module.dequantize_weights)
        + inspect.getsource(embedding_module.QuantizedEmbedding.__call__)
    )
    assert "mx.quantized_matmul" not in source
    assert "mx.dequantize" not in source


def test_task_4_model_integrates_packed_weights_before_fast_kernels():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="quantized-matvec")
    layer = model.layers_inner[0]

    assert isinstance(model.embedding, QuantizedEmbedding)
    assert isinstance(layer.self_attn.wq, QuantizedWeights)
    assert isinstance(layer.mlp.w_gate, QuantizedWeights)
    assert isinstance(layer.input_layernorm, RMSNorm)
    assert isinstance(layer.self_attn.rope, RoPE)
    assert not layer.self_attn.use_decode_attention
    assert not layer.mlp.use_fast_swiglu


def quantized_matmul_helper(
    stream: mx.Stream,
    precision: mx.Dtype,
    identity_matrix: bool,
):
    with mx.stream(stream):
        group_size = 128
        if identity_matrix:
            input = mx.eye(group_size, dtype=precision)
        else:
            input = mx.random.normal(shape=(3, group_size), dtype=precision)
        weight = mx.random.normal(shape=(5, group_size), dtype=precision)
        w_q, scales, biases = mx.quantize(weight, group_size=group_size, bits=4)
        user_out = quantized_matmul(
            scales=scales,
            biases=biases,
            group_size=group_size,
            bits=4,
            a=input,
            b=w_q,
            transpose_b=True,
        )
        ref_out = mx.quantized_matmul(
            input,
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=4,
            transpose=True,
        )
        assert user_out.dtype == mx.bfloat16
        if identity_matrix:
            assert_allclose(user_out, ref_out, precision)
        else:
            assert_allclose(
                user_out,
                ref_out,
                precision,
                atol=5.0e-1,
                message=f"quantized matmul {precision} comparison",
            )


def test_task_3_quantized_matmul_simple_bf16_gpu():
    quantized_matmul_helper(mx.gpu, mx.bfloat16, True)


def test_task_3_quantized_matmul_complex_bf16_gpu():
    quantized_matmul_helper(mx.gpu, mx.bfloat16, False)


def test_task_3_optimized_matvec_matches_vanilla_gpu():
    """The scalar baseline must remain callable for a decode-shaped input."""
    with mx.stream(mx.gpu):
        input = mx.random.normal((1, 256)).astype(mx.bfloat16)
        weight = mx.random.normal((96, 256)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        optimized = quantized_matvec_custom(
            scales, biases, 128, 4, input, packed, transpose_b=True
        )
        vanilla = quantized_matmul_vanilla(
            scales, biases, 128, 4, input, packed, transpose_b=True
        )
        assert_allclose(optimized, vanilla, mx.bfloat16, atol=0.5, rtol=2e-2)


def quantized_matvec_custom_helper(num_rows: int):
    with mx.stream(mx.gpu):
        group_size = 128
        input = mx.random.normal(shape=(num_rows, group_size), dtype=mx.bfloat16)
        weight = mx.random.normal(shape=(64, group_size), dtype=mx.bfloat16)
        w_q, scales, biases = mx.quantize(weight, group_size=group_size, bits=4)
        user_out = quantized_matvec_custom(
            scales=scales,
            biases=biases,
            group_size=group_size,
            bits=4,
            a=input,
            b=w_q,
            transpose_b=True,
        )
        ref_out = mx.quantized_matmul(
            input,
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=4,
            transpose=True,
        )
        assert_allclose(user_out, ref_out, mx.bfloat16, atol=5.0e-1)


def test_task_4_quantized_matvec_custom_m1_gpu():
    quantized_matvec_custom_helper(1)


def test_task_4_quantized_matvec_custom_m8_gpu():
    quantized_matvec_custom_helper(8)


def test_task_4_quantized_matvec_streaming_qwen_shape_gpu():
    with mx.stream(mx.gpu):
        input = mx.random.normal((1, 2560)).astype(mx.bfloat16)
        weight = mx.random.normal((1024, 2560)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        result = quantized_matvec_custom(
            scales, biases, 128, 4, input, packed, transpose_b=True
        )
        expected = mx.quantized_matmul(
            input,
            packed,
            scales,
            biases,
            group_size=128,
            bits=4,
            transpose=True,
        )
        assert_allclose(result, expected, mx.bfloat16, atol=1.5)
