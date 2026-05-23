import torch
from .kv_cache import TinyKvCache
from .qwen3_week2 import (
    Qwen3MLP,
    Qwen3MultiHeadAttention,
    Qwen3TransformerBlock,
)
from typing import Any


class Qwen3ModelWeek3:
    def __init__(
        self,
        mlx_model: Any,
        page_size: int = 128,
    ):
        self.num_hidden_layers = None
        pass

    def create_kv_cache(self) -> list[TinyKvCache]:
        pass

    def __call__(
        self,
        inputs: torch.Tensor,
        offset: int | list[int] | torch.Tensor,
        cache: list[TinyKvCache],
    ) -> torch.Tensor:
        pass
