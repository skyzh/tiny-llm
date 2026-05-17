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


def helper_test_task_3(model_name: str, iters: int = 10):
    mlx_model, tokenizer = load(model_name)
    model = Qwen3ModelWeek2(mlx_model)
    for _ in range(iters):
        cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        input = mx.random.randint(low=0, high=tokenizer.vocab_size, shape=(1, 10))
        user_output = model(input, 0, cache)
        # Component tests cover numerical equivalence; this checks model wiring
        # with quantized weights and a KV cache.
        assert user_output.shape == (1, 10, model.vocab_size)
        assert user_output.dtype == mx.bfloat16
        assert np.all(np.isfinite(np.array(user_output.astype(mx.float32))))


@pytest.mark.skipif(
    not qwen_3_06b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_3_qwen_3_06b():
    helper_test_task_3("Qwen/Qwen3-0.6B-MLX-4bit", 5)


@pytest.mark.skipif(
    not qwen_3_4b_model_exists(), reason="Qwen3-4B-4bit model not found"
)
def test_task_3_qwen_3_4b():
    helper_test_task_3("Qwen/Qwen3-4B-MLX-4bit", 1)


@pytest.mark.skipif(
    not qwen_3_17b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_3_qwen_3_17b():
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
        decode_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        for offset in range(seq_len):
            user_out = model(
                inputs=inputs[:, offset : offset + 1],
                offset=offset,
                cache=decode_cache,
            )
            assert user_out.shape == (1, 1, model.vocab_size)
            assert user_out.dtype == mx.bfloat16
            assert np.all(np.isfinite(np.array(user_out.astype(mx.float32))))


@pytest.mark.skipif(
    not qwen_3_06b_model_exists(), reason="Qwen3-0.6B-4bit model not found"
)
def test_task_4_qwen_3_06b():
    helper_test_task_4("Qwen/Qwen3-0.6B-MLX-4bit", seq_len=3)


@pytest.mark.skipif(
    not qwen_3_4b_model_exists(), reason="Qwen3-4B-4bit model not found"
)
def test_task_4_qwen_3_4b():
    helper_test_task_4(
        "Qwen/Qwen3-4B-MLX-4bit",
        seq_len=3,
    )


@pytest.mark.skipif(
    not qwen_3_17b_model_exists(), reason="Qwen3-1.7B-4bit model not found"
)
def test_task_4_qwen_3_17b():
    helper_test_task_4("Qwen/Qwen3-1.7B-MLX-4bit", seq_len=3)
