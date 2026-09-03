from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from .tiny_llm_base import simple_generate


EOS = 0
VOCAB_SIZE = 8


class FakeDetokenizer:
    def __init__(self):
        self.last_segment = ""
        self.pending = "stale"

    def reset(self):
        self.last_segment = ""
        self.pending = ""

    def add_token(self, token: int):
        self.last_segment = self.pending
        self.pending = {EOS: "<eos>", 2: "A", 3: "B", 4: "C"}.get(token, "?")

    def finalize(self):
        self.last_segment = self.pending
        self.pending = ""


class FakeTokenizer:
    def __init__(self, prompt_tokens: list[int]):
        self.prompt_tokens = prompt_tokens
        self.eos_token_id = EOS
        self.detokenizer = FakeDetokenizer()

    def encode(self, prompt: str, add_special_tokens: bool = True) -> list[int]:
        if add_special_tokens:
            return [6, *self.prompt_tokens]
        return list(self.prompt_tokens)


class ScriptedModel:
    def __init__(self, next_tokens: list[int], decoy_token: int = 6):
        self.next_tokens = list(next_tokens)
        self.decoy_token = decoy_token

    def __call__(self, tokens: mx.array) -> mx.array:
        if not self.next_tokens:
            raise AssertionError("model received an unexpected call")
        next_token = self.next_tokens.pop(0)
        if 6 in np.array(tokens).reshape(-1).tolist():
            next_token = 4
        logits = np.full((1, tokens.shape[1], VOCAB_SIZE), -10_000.0)
        logits[:, :-1, self.decoy_token] = 10_000.0
        logits[:, -1, next_token] = 10_000.0
        return mx.array(logits, dtype=mx.float32)


def test_task_1_greedy_prefill_decode_eos_and_stream_tail(capsys):
    tokenizer = FakeTokenizer([7, EOS])
    model = ScriptedModel([2, 3, EOS])

    result = simple_generate(model, tokenizer, "prompt", sampler=None)

    assert result is None
    assert capsys.readouterr().out == "AB"


def test_task_1_custom_sampler_receives_stable_log_probabilities(capsys):
    tokenizer = FakeTokenizer([5])
    model = ScriptedModel([2, EOS])
    sampler_inputs: list[mx.array] = []
    sampled_tokens = iter([4, EOS])

    def sampler(logprobs: mx.array) -> mx.array:
        sampler_inputs.append(logprobs)
        return mx.array([next(sampled_tokens)], dtype=mx.int32)

    simple_generate(model, tokenizer, "prompt", sampler=sampler)

    assert sampler_inputs
    for logprobs in sampler_inputs:
        values = np.array(logprobs)
        assert values.shape == (1, VOCAB_SIZE)
        assert np.isfinite(values).all()
        np.testing.assert_allclose(np.exp(values).sum(axis=-1), [1.0], atol=1e-6)
    assert capsys.readouterr().out == "C"


def test_task_1_max_tokens_bounds_non_eos_generation(capsys):
    tokenizer = FakeTokenizer([5])
    model = ScriptedModel([2, 2, 2, 2])

    simple_generate(model, tokenizer, "prompt", sampler=None, max_tokens=3)

    assert capsys.readouterr().out == "AAA"


def test_task_1_rejects_empty_encoding():
    tokenizer = FakeTokenizer([])
    model = ScriptedModel([EOS])

    with pytest.raises(Exception):
        simple_generate(model, tokenizer, "prompt", sampler=None)
