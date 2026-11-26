"""Exercise the supplied CLI through the Week 2 dispatch and generation helper."""

import json

import huggingface_hub
import mlx.core as mx
import mlx_lm
import pytest

import agent as cli
import tiny_llm
from tiny_llm_ref import agent, models
from tiny_llm_ref.kv_cache import TinyKvFullCache
from week4_fixture import _APP

_MODEL = "Qwen/Qwen3-0.6B-MLX-4bit"
_READ = {"tool": "read_file", "path": "app.py"}


def _backend(monkeypatch, model_path, responses):
    prompts, caches, offsets, downloads, loads = [], [], [], [], []
    responses = iter(responses)
    loaded_weights = object()

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **options):
            assert options == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            prompts.append(tuple(message.copy() for message in messages))
            return "chat"

        def encode(self, prompt, *, add_special_tokens):
            assert prompt == "chat" and not add_special_tokens
            return [2] * 257

        def decode(self, tokens):
            assert tokens == [1]
            return json.dumps(next(responses))

    class WeekTwo:
        def __init__(self, weights, *, checkpoint, use_mlx_quantized_linear):
            assert weights is loaded_weights
            assert checkpoint == "packed-weights" and use_mlx_quantized_linear

        def create_kv_cache(self):
            cache = TinyKvFullCache()
            assert (
                cache.capacity is None and cache.offset == cache.allocated_capacity == 0
            )
            caches.append(cache)
            return [cache]

        def __call__(self, tokens, offset, active):
            (cache,) = active
            assert cache is caches[-1] and cache.offset == offset
            offsets.append(offset)
            keys = mx.zeros((1, 1, tokens.size, 4))
            cache.update_and_fetch(keys, keys)
            assert cache.allocated_capacity == 512
            return mx.array([[[0, 1]]] if offset == 0 else [[[1, 0]]])

    def download(model_name, *, local_files_only):
        downloads.append((model_name, local_files_only))
        return str(model_path)

    def load(path, *, tokenizer_config):
        assert tokenizer_config == {"trust_remote_code": False}
        loads.append(path)
        return loaded_weights, Tokenizer()

    def wrong_generation(*_args, **_kwargs):
        pytest.fail("the CLI must generate through the course model")

    monkeypatch.setattr(tiny_llm, "agent", agent)
    monkeypatch.setattr(tiny_llm, "models", models)
    monkeypatch.setattr(models, "Qwen3ModelWeek2", WeekTwo)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(mlx_lm, "load", load)
    monkeypatch.setattr(mlx_lm, "generate", wrong_generation)
    return prompts, caches, offsets, downloads, loads


def _options():
    return ["--model", "qwen3-0.6b", "--device", "cpu"]


@pytest.mark.parametrize("inspect", [True, False])
def test_cli_reaches_week_two_and_observes_the_read(
    inspect, monkeypatch, tmp_path, capsys
):
    model_path = tmp_path / "cached"
    prompts, caches, offsets, downloads, loads = _backend(
        monkeypatch, model_path, [_READ, {"final": "It returns a greeting."}]
    )
    if inspect:
        args = ["--inspect-fixture"]
    else:
        (tmp_path / "app.py").write_bytes(_APP)
        args = ["--root", str(tmp_path), "Read the greeting."]
    assert cli.main(_options() + args) == 0
    output = capsys.readouterr().out
    implementation = "reference" if inspect else "learner"
    assert f"backend> Week 2 {implementation}" in output
    assert "observation> " + _APP.decode() in output
    assert "finished> It returns a greeting." in output
    assert ("fixture> observed the expected app.py bytes" in output) is inspect
    assert downloads == ([(_MODEL, True)] if inspect else [])
    assert loads == [str(model_path) if inspect else _MODEL]
    assert prompts[1][-1] == {
        "role": "user",
        "content": "Tool result:\n" + _APP.decode(),
    }
    if inspect:
        assert prompts[0][0]["content"].endswith('{"tool":"read_file","path":"app.py"}')
        assert "Use the tool result" in prompts[1][0]["content"]
    assert len(caches) == 2 and caches[0] is not caches[1]
    assert offsets == [0, 257, 0, 257]


def test_cli_cannot_finish_the_inspection_without_a_read(monkeypatch, tmp_path, capsys):
    prompts, caches, offsets, _, _ = _backend(
        monkeypatch, tmp_path, [{"final": "I finished."}]
    )
    assert cli.main(_options() + ["--inspect-fixture", "--max-steps", "1"]) == 1
    output = capsys.readouterr().out
    assert "stopped> the required read of app.py was not observed" in output
    assert "finished>" not in output
    assert len(prompts) == len(caches) == 1
    assert offsets == [0, 257]


def test_cli_reports_a_missing_local_snapshot_before_model_loading(monkeypatch, capsys):
    def missing(*_args, **_kwargs):
        from huggingface_hub.errors import LocalEntryNotFoundError

        raise LocalEntryNotFoundError("missing test snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(SystemExit) as error:
        cli.main(_options() + ["--inspect-fixture"])
    assert error.value.code == 1
    assert "missing from the local model cache" in capsys.readouterr().err
