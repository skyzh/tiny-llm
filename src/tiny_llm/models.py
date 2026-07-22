from model_names import shortcut_name_to_full_name

from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from .qwen3_week3 import Qwen3ModelWeek3


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
