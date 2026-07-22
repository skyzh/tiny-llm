from collections.abc import Callable, Sequence
from typing import Any


Message = dict[str, str]
Generate = Callable[[list[Message]], str]


def initial_messages(task: str, system_prompt: str) -> list[Message]:
    """Week 4, Day 1: create the durable beginning of an agent conversation."""

    if not task.strip():
        raise ValueError("task must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]


def generate_response(
    model,
    tokenizer,
    messages: list[Message],
    cache_factory: Callable[[], Sequence[Any]],
    max_tokens: int,
    enable_thinking: bool = False,
) -> str:
    """Week 4, Day 1: decode one action with the course model and a fresh cache."""

    import mlx.core as mx

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    caches = list(cache_factory())
    output: list[int] = []
    offset = 0
    try:
        for _ in range(max_tokens):
            logits = model(tokens[None], offset, caches)[:, -1, :]
            token = int(mx.argmax(logits, axis=-1).item())
            if token == tokenizer.eos_token_id:
                break
            output.append(token)
            offset += tokens.size
            tokens = mx.array([token])
        return tokenizer.decode(output)
    finally:
        for cache in caches:
            release = getattr(cache, "release", None)
            if release is not None:
                release()
