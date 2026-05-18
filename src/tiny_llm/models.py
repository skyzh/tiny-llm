from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from .qwen3_week3 import Qwen3ModelWeek3


def shortcut_name_to_full_name(shortcut_name: str):
    lower_shortcut_name = shortcut_name.lower()
    if lower_shortcut_name == "qwen3-8b":
        return "Qwen/Qwen3-8B-MLX-4bit"
    elif lower_shortcut_name == "qwen3-0.6b":
        return "Qwen/Qwen3-0.6B-MLX-4bit"
    elif lower_shortcut_name == "qwen3-1.7b":
        return "Qwen/Qwen3-1.7B-MLX-4bit"
    elif lower_shortcut_name == "qwen3-4b":
        return "Qwen/Qwen3-4B-MLX-4bit"
    else:
        return shortcut_name


def dispatch_model(model_name: str, mlx_model, week: int, **kwargs):
    model_name = shortcut_name_to_full_name(model_name)
    is_qwen3 = model_name.startswith("Qwen/Qwen3")
    if week == 1 and is_qwen3:
        return Qwen3ModelWeek1(mlx_model, **kwargs)
    elif week == 2 and is_qwen3:
        return Qwen3ModelWeek2(mlx_model, **kwargs)
    elif week == 3 and is_qwen3:
        return Qwen3ModelWeek3(mlx_model, **kwargs)
    else:
        raise ValueError(f"{model_name} for week {week} not supported")
