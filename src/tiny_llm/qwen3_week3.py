import mlx.core as mx
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
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        pass

    def create_kv_cache(self) -> list[TinyKvCache]:
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int | list[int] | mx.array,
        cache: list[TinyKvCache],
    ) -> mx.array:
        pass
