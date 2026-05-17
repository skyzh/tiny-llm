import pytest
from .utils import *
from .tiny_llm_base import (
    Qwen3ModelWeek2,
    TinyKvFullCache,
)
from mlx_lm import load

# TODO: task 1 tests


@pytest.mark.skipif(
    not qwen_3_06b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_utils_qwen_3_06b():
    pass


@pytest.mark.skipif(
    not qwen_3_4b_model_exists(), reason="Qwen3-4B-4bit model not found"
)
def test_utils_qwen_3_4b():
    pass


@pytest.mark.skipif(
    not qwen_3_17b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_utils_qwen_3_17b():
    pass


def helper_test_task_3(
    model_name: str,
    iters: int = 10,
    rtol: float = 0.1,
    atol: float = 0.5,
    max_allowed_mismatches: int = 0,
):
    mlx_model, tokenizer = load(model_name)
    model = Qwen3ModelWeek2(mlx_model)
    for _ in range(iters):
        cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        input = mx.random.randint(low=0, high=tokenizer.vocab_size, shape=(1, 10))
        user_output = model(input, 0, cache)
        user_output = user_output - mx.logsumexp(user_output, keepdims=True)
        ref_output = mlx_model(input)
        ref_output = ref_output - mx.logsumexp(ref_output, keepdims=True)
        assert_allclose(
            user_output,
            ref_output,
            precision=mx.bfloat16,
            rtol=rtol,
            atol=atol,
            max_allowed_mismatches=max_allowed_mismatches,
        )


@pytest.mark.skipif(
    not qwen_3_06b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_3_qwen_3_06b():
    helper_test_task_3("Qwen/Qwen3-0.6B-MLX-4bit", 5)


@pytest.mark.skipif(
    not qwen_3_4b_model_exists(), reason="Qwen3-4B-4bit model not found"
)
def test_task_3_qwen_3_4b():
    helper_test_task_3(
        "Qwen/Qwen3-4B-MLX-4bit",
        1,
        rtol=2.5e-1,
        atol=5e-1,
        max_allowed_mismatches=1024,
    )


@pytest.mark.skipif(
    not qwen_3_17b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_3_qwen_3_17b():
    helper_test_task_3("Qwen/Qwen3-1.7B-MLX-4bit", 3)


def helper_test_task_4(
    model_name: str,
    seq_len: int,
    iters: int = 1,
    rtol: float = 1e-1,
    atol: float | None = None,
    max_allowed_mismatches: int = 0,
):
    mlx_model, tokenizer = load(model_name)
    model = Qwen3ModelWeek2(mlx_model)
    for _ in range(iters):
        cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        inputs = mx.random.randint(0, tokenizer.vocab_size, (1, seq_len))
        ref_outputs = mlx_model(inputs)
        for offset in range(seq_len):
            user_out = model(
                inputs=inputs[:, offset : offset + 1], offset=offset, cache=cache
            )
            ref_out = ref_outputs[:, offset : offset + 1, :]
            user_out = user_out - mx.logsumexp(user_out, keepdims=True)
            ref_out = ref_out - mx.logsumexp(ref_out, keepdims=True)
            assert_allclose(
                user_out,
                ref_out,
                precision=mx.bfloat16,
                rtol=rtol,
                atol=atol,
                max_allowed_mismatches=max_allowed_mismatches,
            )


@pytest.mark.skipif(
    not qwen_3_06b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_4_qwen_3_06b():
    helper_test_task_4("Qwen/Qwen3-0.6B-MLX-4bit", seq_len=3, max_allowed_mismatches=32)


@pytest.mark.skipif(
    not qwen_3_4b_model_exists(), reason="Qwen3-4B-4bit model not found"
)
def test_task_4_qwen_3_4b():
    helper_test_task_4(
        "Qwen/Qwen3-4B-MLX-4bit",
        seq_len=3,
        rtol=2.5e-1,
        atol=5e-1,
        max_allowed_mismatches=1024,
    )


@pytest.mark.skipif(
    not qwen_3_17b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_4_qwen_3_17b():
    helper_test_task_4("Qwen/Qwen3-1.7B-MLX-4bit", seq_len=3, max_allowed_mismatches=32)
