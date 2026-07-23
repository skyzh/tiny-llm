"""Week 2 Day 6 SIMD-matrix prefill tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import (
    QuantizedEmbedding,
    QuantizedWeights,
    Qwen3ModelWeek2,
    quantized_matmul,
    quantized_matmul_vanilla,
)
from .utils import (
    assert_allclose,
    qwen3_0_6b_model_exists,
    qwen3_1_7b_model_exists,
    qwen3_4b_model_exists,
    tiny_qwen3_mlx_model,
)
from mlx_lm import load


def test_simd_matmul_checkpoint_is_completed_week2_model():
    model = Qwen3ModelWeek2(tiny_qwen3_mlx_model(), checkpoint="simd-matmul")
    layer = model.layers_inner[0]

    assert model.embedding.use_custom_kernel
    assert model.embedding.weight.use_simdgroup_matmul
    assert layer.self_attn.wq.use_simdgroup_matmul


def test_task_2_simdgroup_matmul_matches_vanilla_gpu():
    with mx.stream(mx.gpu):
        inputs = mx.random.normal((128, 256)).astype(mx.bfloat16)
        weight = mx.random.normal((96, 256)).astype(mx.bfloat16)
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
        vanilla = quantized_matmul_vanilla(
            scales, biases, 128, 4, inputs, packed, transpose_b=True
        )
        assert_allclose(tiled, vanilla, mx.bfloat16, atol=1.0, rtol=2e-2)


def test_task_2_simdgroup_matmul_uses_accurate_partial_tiles_gpu():
    """A non-multiple-of-eight prefill must not accumulate in bfloat16."""
    with mx.stream(mx.gpu):
        inputs = mx.random.normal((10, 256)).astype(mx.bfloat16)
        weight = mx.random.normal((96, 256)).astype(mx.bfloat16)
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
        vanilla = quantized_matmul_vanilla(
            scales, biases, 128, 4, inputs, packed, transpose_b=True
        )
        assert_allclose(tiled, vanilla, mx.bfloat16, atol=0.25, rtol=1e-2)


def test_task_4_custom_embedding_matches_readable_path():
    weight = mx.random.normal((17, 256)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
    quantized = QuantizedWeights(scales, biases, 128, 4, packed)
    readable = QuantizedEmbedding(17, 256, quantized)
    custom = QuantizedEmbedding(17, 256, quantized, use_custom_kernel=True)
    indices = mx.array([[1, 4, 9]], dtype=mx.int32)
    assert_allclose(
        custom(indices),
        readable(indices),
        mx.bfloat16,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(
    not qwen3_0_6b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_utils_qwen3_0_6b():
    pass


@pytest.mark.skipif(not qwen3_4b_model_exists(), reason="Qwen3-4B-4bit model not found")
def test_utils_qwen3_4b():
    pass


@pytest.mark.skipif(
    not qwen3_1_7b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_utils_qwen3_1_7b():
    pass


def helper_test_task_6(model_name: str, iters: int = 10):
    mlx_model, tokenizer = load(model_name)
    model = Qwen3ModelWeek2(mlx_model, checkpoint="simd-matmul")
    assert model.embedding.use_custom_kernel
    assert model.embedding.weight.use_simdgroup_matmul
    assert all(layer.self_attn.wq.use_simdgroup_matmul for layer in model.layers_inner)
    for iteration in range(iters):
        cache = model.create_kv_cache()
        input = (mx.arange(10, dtype=mx.int32) + iteration * 10).reshape(
            1, 10
        ) % tokenizer.vocab_size
        user_output = model(input, 0, cache)
        ref_output = mlx_model(input)
        user_output = user_output - mx.logsumexp(user_output, axis=-1, keepdims=True)
        ref_output = ref_output - mx.logsumexp(ref_output, axis=-1, keepdims=True)
        assert_allclose(
            user_output, ref_output, precision=mx.bfloat16, rtol=0.1, atol=2.5
        )


@pytest.mark.skipif(
    not qwen3_0_6b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_6_qwen3_0_6b():
    helper_test_task_6("Qwen/Qwen3-0.6B-MLX-4bit", 5)


@pytest.mark.skipif(not qwen3_4b_model_exists(), reason="Qwen3-4B-4bit model not found")
def test_task_6_qwen3_4b():
    helper_test_task_6("Qwen/Qwen3-4B-MLX-4bit", 1)


@pytest.mark.skipif(
    not qwen3_1_7b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_6_qwen3_1_7b():
    helper_test_task_6("Qwen/Qwen3-1.7B-MLX-4bit", 3)


def helper_test_task_6_incremental(
    model_name: str,
    seq_len: int,
    iters: int = 1,
):
    mlx_model, tokenizer = load(model_name)
    model = Qwen3ModelWeek2(mlx_model, checkpoint="simd-matmul")
    for _ in range(iters):
        inputs = mx.random.randint(0, tokenizer.vocab_size, (1, seq_len))
        ref_outputs = mlx_model(inputs)
        decode_cache = model.create_kv_cache()
        for offset in range(seq_len):
            user_out = model(
                inputs=inputs[:, offset : offset + 1],
                offset=offset,
                cache=decode_cache,
            )
            ref_out = ref_outputs[:, offset : offset + 1, :]
            user_out = user_out - mx.logsumexp(user_out, axis=-1, keepdims=True)
            ref_out = ref_out - mx.logsumexp(ref_out, axis=-1, keepdims=True)
            assert_allclose(
                user_out, ref_out, precision=mx.bfloat16, rtol=0.1, atol=2.5
            )


@pytest.mark.skipif(
    not qwen3_0_6b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_6_incremental_qwen3_0_6b():
    helper_test_task_6_incremental("Qwen/Qwen3-0.6B-MLX-4bit", seq_len=3)


@pytest.mark.skipif(not qwen3_4b_model_exists(), reason="Qwen3-4B-4bit model not found")
def test_task_6_incremental_qwen3_4b():
    helper_test_task_6_incremental(
        "Qwen/Qwen3-4B-MLX-4bit",
        seq_len=3,
    )


@pytest.mark.skipif(
    not qwen3_1_7b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_6_incremental_qwen3_1_7b():
    helper_test_task_6_incremental("Qwen/Qwen3-1.7B-MLX-4bit", seq_len=3)


class FakeEmbedding:
    def __call__(self, inputs):
        return mx.stack([inputs, inputs + 1], axis=-1).astype(mx.bfloat16)

    def as_linear(self, hidden):
        return hidden


@pytest.mark.parametrize("logits_to_keep,expected_length", [(1, 1), (None, 4)])
def test_task_5_logits_to_keep_controls_output_length(logits_to_keep, expected_length):
    model = Qwen3ModelWeek2.__new__(Qwen3ModelWeek2)
    model.num_hidden_layers = 0
    model.embedding = FakeEmbedding()
    model.layers_inner = []
    model.norm = lambda hidden: hidden
    model.w_lm_head = None
    inputs = mx.array([[1, 2, 3, 4]])
    result = model(inputs, 0, [], logits_to_keep=logits_to_keep)
    assert result.shape == (1, expected_length, 2)
