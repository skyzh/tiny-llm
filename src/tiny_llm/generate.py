import torch
from typing import Any, Callable

from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: Any,
    prompt: str,
    sampler: Callable[[torch.Tensor], torch.Tensor] | None,
) -> str:
    def _step(model, y):
        pass


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2, tokenizer: Any, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: Any,
    tokenizer: Any,
    prompt: str,
) -> str:
    pass
