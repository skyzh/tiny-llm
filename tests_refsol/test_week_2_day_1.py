import pytest
from .utils import *
from .tiny_llm_base import (
    Qwen3ModelWeek2,
    TinyKvFullCache,
)
from mlx_lm import load

# TODO: task 1 tests


@pytest.mark.skipif(
    not qwen3_0_6b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_utils_qwen3_0_6b():
    pass


@pytest.mark.skipif(
    not qwen3_4b_model_exists(), reason="Qwen3-4B-4bit model not found"
)
def test_utils_qwen3_4b():
    pass


@pytest.mark.skipif(
    not qwen3_1_7b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_utils_qwen3_1_7b():
    pass


def helper_test_task_3(model_name: str, iters: int = 10):
    mlx_model, tokenizer = load(model_name)
    model = Qwen3ModelWeek2(mlx_model)
    for _ in range(iters):
        cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        input = mx.random.randint(low=0, high=tokenizer.vocab_size, shape=(1, 10))
        user_output = model(input, 0, cache)
        ref_output = mlx_model(input)
        user_output = user_output - mx.logsumexp(user_output, keepdims=True)
        ref_output = ref_output - mx.logsumexp(ref_output, keepdims=True)
        assert_allclose(
            user_output, ref_output, precision=mx.bfloat16, rtol=0.1, atol=1.0
        )


@pytest.mark.skipif(
    not qwen3_0_6b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_3_qwen3_0_6b():
    helper_test_task_3("Qwen/Qwen3-0.6B-MLX-4bit", 5)


@pytest.mark.skipif(
    not qwen3_4b_model_exists(), reason="Qwen3-4B-4bit model not found"
)
def test_task_3_qwen3_4b():
    helper_test_task_3("Qwen/Qwen3-4B-MLX-4bit", 1)


@pytest.mark.skipif(
    not qwen3_1_7b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_3_qwen3_1_7b():
    helper_test_task_3("Qwen/Qwen3-1.7B-MLX-4bit", 3)


def helper_test_task_4(
    model_name: str,
    seq_len: int,
    iters: int = 1,
):
    mlx_model, tokenizer = load(model_name)
    model = Qwen3ModelWeek2(mlx_model)
    for _ in range(iters):
        inputs = mx.random.randint(0, tokenizer.vocab_size, (1, seq_len))
        ref_outputs = mlx_model(inputs)
        decode_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        for offset in range(seq_len):
            user_out = model(
                inputs=inputs[:, offset : offset + 1],
                offset=offset,
                cache=decode_cache,
            )
            ref_out = ref_outputs[:, offset : offset + 1, :]
            user_out = user_out - mx.logsumexp(user_out, keepdims=True)
            ref_out = ref_out - mx.logsumexp(ref_out, keepdims=True)
            assert_allclose(
                user_out, ref_out, precision=mx.bfloat16, rtol=0.1, atol=1.0
            )


@pytest.mark.skipif(
    not qwen3_0_6b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_4_qwen3_0_6b():
    helper_test_task_4("Qwen/Qwen3-0.6B-MLX-4bit", seq_len=3)


@pytest.mark.skipif(
    not qwen3_4b_model_exists(), reason="Qwen3-4B-4bit model not found"
)
def test_task_4_qwen3_4b():
    helper_test_task_4(
        "Qwen/Qwen3-4B-MLX-4bit",
        seq_len=3,
    )


@pytest.mark.skipif(
    not qwen3_1_7b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_4_qwen3_1_7b():
    helper_test_task_4("Qwen/Qwen3-1.7B-MLX-4bit", seq_len=3)
