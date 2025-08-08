import pytest
from .utils import *
from .tiny_llm_base import Qwen3ModelWeek1, Embedding, dequantize_linear, qwen3_week1
from mlx_lm import load

# TODO: task 1 tests


@pytest.mark.skipif(
    not qwen_3_06b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_utils_qwen_3_06b():
    pass


@pytest.mark.skipif(
    not qwen_3_8b_model_exists(), reason="Qwen3-8B-4bit model not found"
)
def test_utils_qwen_3_8b():
    pass


@pytest.mark.skipif(
    not qwen_3_17b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_utils_qwen_3_17b():
    pass


def helper_test_task_3(model_name: str, iters: int = 10):
    mlx_model, tokenizer = load(model_name)
    force_convert_bf16_to(mlx_model, mx.float16)
    model = Qwen3ModelWeek1(mlx_model)
    for _ in range(iters):
        input = mx.random.randint(low=0, high=tokenizer.vocab_size, shape=(1, 10))
        user_output = model(input, 0)
        user_output = user_output - mx.logsumexp(user_output, keepdims=True)
        ref_output = mlx_model(input)
        ref_output = ref_output - mx.logsumexp(ref_output, keepdims=True)
        assert_allclose(
            user_output, ref_output, precision=mx.float16, atol=1, rtol=2e-1
        )


@pytest.mark.skipif(
    not qwen_3_06b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_2_embedding_call():
    mlx_model, _ = load("mlx-community/Qwen3-0.6B-4bit")
    embedding = Embedding(
        mlx_model.args.vocab_size,
        mlx_model.args.hidden_size,
        dequantize_linear(mlx_model.model.embed_tokens).astype(mx.float16),
    )
    force_convert_bf16_to(mlx_model, mx.float16)
    for _ in range(50):
        input = mx.random.randint(low=0, high=mlx_model.args.vocab_size, shape=(1, 10))
        user_output = embedding(input)
        ref_output = mlx_model.model.embed_tokens(input)
        assert_allclose(user_output, ref_output, precision=mx.float16)


@pytest.mark.skipif(
    not qwen_3_06b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_2_embedding_as_linear():
    mlx_model, _ = load("mlx-community/Qwen3-0.6B-4bit")
    embedding = Embedding(
        mlx_model.args.vocab_size,
        mlx_model.args.hidden_size,
        dequantize_linear(mlx_model.model.embed_tokens).astype(mx.float16),
    )
    for _ in range(50):
        input = mx.random.uniform(shape=(1, 10, mlx_model.args.hidden_size))
        user_output = embedding.as_linear(input)
        ref_output = mlx_model.model.embed_tokens.as_linear(input)
        assert_allclose(user_output, ref_output, precision=mx.float16, atol=1e-1)


@pytest.mark.skipif(
    not qwen_3_06b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_3_qwen_3_06b():
    helper_test_task_3("mlx-community/Qwen3-0.6B-4bit", 5)


@pytest.mark.skipif(
    not qwen_3_8b_model_exists(), reason="Qwen3-8B-4bit model not found"
)
def test_task_3_qwen_3_8b():
    helper_test_task_3("mlx-community/Qwen3-8B-4bit", 1)


@pytest.mark.skipif(
    not qwen_3_17b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_3_qwen_3_17b():
    helper_test_task_3("mlx-community/Qwen3-1.7B-4bit", 3)
