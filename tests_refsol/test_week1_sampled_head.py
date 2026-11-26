import mlx.core as mx
import pytest

from extensions_ref import tiny_llm_ext_ref
from .tiny_llm_base import simple_generate
from tiny_llm_ref.embedding import Embedding
from tiny_llm_ref.layer_norm import RMSNorm
from tiny_llm_ref.qwen3_week1 import Qwen3ModelWeek1


class Tokenizer:
    eos_token_id = -1

    def __init__(self, length):
        self.length = length
        self.detokenizer = self

    def encode(self, prompt, add_special_tokens=False):
        return [index % 5 for index in range(self.length)]

    def reset(self):
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)

    def finalize(self):
        pass

    last_segment = ""


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("available", [True, False])
def test_week1_sampled_head_is_used_only_for_generation(monkeypatch, tied, available):
    model = Qwen3ModelWeek1.__new__(Qwen3ModelWeek1)
    model.embedding = Embedding(5, 3, mx.arange(15).reshape(5, 3).astype(mx.bfloat16))
    model.norm = RMSNorm(3, mx.ones((3,), dtype=mx.bfloat16))
    model.num_hidden_layers = 0
    model.w_lm_head = None if tied else -model.embedding.weight
    expected_weight = model.embedding.weight if tied else model.w_lm_head
    calls = []

    def sampled_head(normalized, weight):
        calls.append(normalized.shape)
        assert weight is expected_weight
        if available:
            return (normalized @ weight.T)[:, -1:, :]
        return None

    monkeypatch.setattr(
        tiny_llm_ext_ref._ext, "_try_week1_sampled_head", sampled_head, raising=False
    )
    for length in (127, 128, 129):
        tokenizer = Tokenizer(length)
        inputs = mx.array(tokenizer.encode("prompt"))[None]
        logits = model(inputs)
        assert logits.shape == (1, length, 5)
        row = logits[:, -1, :]
        expected = mx.argmax(row - mx.logsumexp(row, keepdims=True), axis=-1).item()
        simple_generate(model, tokenizer, "prompt", sampler=None, max_tokens=1)
        assert tokenizer.tokens == [expected]

    assert calls == [(1, 128, 3)]
