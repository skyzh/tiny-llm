import mlx.core as mx
import pytest

from tiny_llm_ref.generate import simple_generate_with_kv_cache
from tiny_llm_ref.kv_cache import TinyKvFullCache
from tiny_llm_ref.qwen3_week2 import (
    Qwen3ModelWeek2,
    Qwen3MultiHeadAttention,
    Qwen3TransformerBlock,
)


class _Tokenizer:
    eos_token_id = 0
    last_segment = ""

    def __init__(self, length):
        self.length = length
        self.detokenizer = self
        self.emitted = []

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [10] * self.length

    def reset(self):
        self.emitted.clear()

    def add_token(self, token):
        self.emitted.append(token)


def _model():
    model = object.__new__(Qwen3ModelWeek2)
    model.checkpoint = "selected"
    model.precision = mx.bfloat16
    model.num_hidden_layers = 36
    model.use_bounded_kv_capacity = False
    model.layers_inner = []
    for _ in range(model.num_hidden_layers):
        layer = object.__new__(Qwen3TransformerBlock)
        attention = object.__new__(Qwen3MultiHeadAttention)
        attention.use_tiled_prefill_attention = True
        layer.self_attn = attention
        model.layers_inner.append(layer)
    return model


@pytest.mark.parametrize(
    "length,cap,bounded,override,stop,marked",
    [
        (251, 1, None, False, "token", True),
        (4092, 1, None, False, "eos", True),
        (251, 1, None, False, "error", True),
        (256, 1, None, False, "token", False),
        (251, 0, None, False, "token", False),
        (251, 2, None, False, "token", False),
        (251, 1, True, False, "token", False),
        (251, 1, False, False, "token", False),
        (251, 1, None, True, "token", False),
    ],
)
def test_cap_one_marking_and_release(
    monkeypatch, length, cap, bounded, override, stop, marked
):
    model = _model()
    tokenizer = _Tokenizer(length)
    calls, released = [], []
    if override:
        factory = model.create_kv_cache
        monkeypatch.setattr(
            model, "create_kv_cache", lambda capacity=None: factory(capacity)
        )

    def prefill(instance, ids, offset, cache, *, logits_to_keep):
        assert instance is model and logits_to_keep == 1
        calls.append(ids.size)
        assert all(
            getattr(owner, "_disposable_prefill_length", None)
            == (length if marked else None)
            for owner in cache
        )
        if stop == "error":
            raise RuntimeError("prefill failed")
        return mx.array([[[1.0, 0.0] if stop == "eos" else [0.0, 1.0]]])

    def release(owner):
        assert not hasattr(owner, "_disposable_prefill_length")
        released.append(owner)

    monkeypatch.setattr(Qwen3ModelWeek2, "__call__", prefill)
    monkeypatch.setattr(TinyKvFullCache, "release", release)

    if stop == "error":
        with pytest.raises(RuntimeError, match="prefill failed"):
            simple_generate_with_kv_cache(model, tokenizer, "fixture", cap, bounded)
    else:
        simple_generate_with_kv_cache(model, tokenizer, "fixture", cap, bounded)

    assert len(released) == (36 if cap else 0)
    assert calls == ([] if cap == 0 else [length, 1] if cap == 2 else [length])
    assert len(tokenizer.emitted) == (cap if stop == "token" else 0)
