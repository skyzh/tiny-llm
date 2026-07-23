MODEL_SHORTCUTS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-MLX-4bit",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B-MLX-4bit",
    "qwen3-4b": "Qwen/Qwen3-4B-MLX-4bit",
    "qwen3-8b": "Qwen/Qwen3-8B-MLX-4bit",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-MLX-4bit",
    "qwen3-moe-30b-a3b": "Qwen/Qwen3-30B-A3B-MLX-4bit",
}


def shortcut_name_to_full_name(model_name: str) -> str:
    return MODEL_SHORTCUTS.get(model_name.lower(), model_name)
