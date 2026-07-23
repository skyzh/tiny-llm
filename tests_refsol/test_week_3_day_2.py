"""Week 3 Day 2 chunked-prefill tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import Request, TinyKvFullCache


class FakeDetokenizer:
    def __init__(self, _):
        self.text = ""

    def add_token(self, token):
        self.text += str(token)


class FakeTokenizer:
    eos_token_id = 99
    _tokenizer = object()
    detokenizer = FakeDetokenizer(_tokenizer)

    def encode(self, prompt, add_special_tokens=False):
        assert not add_special_tokens
        return list(range(1, len(prompt) + 1))


class FakeModel:
    num_hidden_layers = 1

    def __init__(self):
        self.calls = []

    def create_kv_cache(self):
        return [TinyKvFullCache()]

    def __call__(self, inputs, offsets, cache, logits_to_keep=1):
        offset = offsets[0] if isinstance(offsets, list) else int(offsets)
        self.calls.append((offset, inputs.shape[1]))
        length = inputs.shape[1]
        key = mx.zeros((1, 1, length, 1), dtype=mx.float32)
        cache[0].update_and_fetch(key, key)
        logits = mx.zeros((1, 1, 4), dtype=mx.float32)
        return logits.at[..., 1].add(1)


def test_chunked_prefill_bounds_work_and_advances_cache():
    model = FakeModel()
    request = Request(model, FakeTokenizer(), "1234567", prefill_max_step=3)

    request.try_prefill()
    assert request.offset == 3
    assert request.kv_cache[0].offset == 3
    assert not request.is_prefill_done

    request.try_prefill()
    assert request.offset == 6
    assert request.kv_cache[0].offset == 6
    assert not request.is_prefill_done

    request.try_prefill()
    assert request.offset == 7
    assert request.kv_cache[0].offset == 7
    assert request.is_prefill_done
    assert request.next_token == 1
    assert model.calls == [(0, 3), (3, 3), (6, 1)]

    with pytest.raises(ValueError, match="after done"):
        request.try_prefill()
