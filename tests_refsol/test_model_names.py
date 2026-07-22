import pytest

from model_names import shortcut_name_to_full_name


@pytest.mark.parametrize(
    ("shortcut", "full_name"),
    [
        ("qwen3-0.6b", "Qwen/Qwen3-0.6B-MLX-4bit"),
        ("qwen3-1.7b", "Qwen/Qwen3-1.7B-MLX-4bit"),
        ("QWEN3-4B", "Qwen/Qwen3-4B-MLX-4bit"),
        ("qwen3-8b", "Qwen/Qwen3-8B-MLX-4bit"),
        ("qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B-MLX-4bit"),
        ("qwen3-moe-30b-a3b", "Qwen/Qwen3-30B-A3B-MLX-4bit"),
    ],
)
def test_shortcut_name_to_full_name(shortcut, full_name):
    assert shortcut_name_to_full_name(shortcut) == full_name


def test_full_model_name_is_unchanged():
    model_name = "some-org/custom-model"
    assert shortcut_name_to_full_name(model_name) == model_name
