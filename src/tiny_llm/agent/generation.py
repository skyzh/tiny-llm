from collections.abc import Callable, Sequence
from typing import Any


Message = dict[str, str]
Generate = Callable[[list[Message]], str]


def initial_messages(task: str, system_prompt: str) -> list[Message]:
    """Week 4, Day 1: create the durable beginning of an agent conversation."""

    pass


def generate_response(
    model,
    tokenizer,
    messages: list[Message],
    cache_factory: Callable[[], Sequence[Any]],
    max_tokens: int,
    enable_thinking: bool = False,
) -> str:
    """Week 4, Day 1: decode one action with the course model and a fresh cache."""

    pass
