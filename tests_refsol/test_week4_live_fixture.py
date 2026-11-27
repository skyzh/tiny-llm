"""Check the live fixture orchestration with the course cache on a tiny model."""

import hashlib
import json
from tempfile import TemporaryDirectory

import mlx.core as mx
import pytest

import agent as cli
import week4_live_fixture as live
from tiny_llm_ref import agent as api
from tiny_llm_ref.kv_cache import TinyKvFullCache
from week4_fixture import _VALIDATE, _WRITE


class TinyModel:
    num_hidden_layers = 1

    def create_kv_cache(self):
        return [TinyKvFullCache()]

    def __call__(self, tokens, offset, caches, *, logits_to_keep=None):
        (cache,) = caches
        assert offset == cache.offset
        values = mx.zeros((1, 1, tokens.size, 2))
        cache.update_and_fetch(values, values)
        return mx.array([[[0, 1]]] if tokens.size > 1 else [[[1, 0]]])


class Tokenizer:
    eos_token_id = 0

    def __init__(self, responses):
        self.responses = iter(responses)

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert not tokenize and not enable_thinking
        text = "".join(f"{entry['role']}: {entry['content']}\n" for entry in messages)
        return text + ("assistant: " if add_generation_prompt else "")

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(map(ord, text))

    def decode(self, tokens):
        assert tokens == [1]
        return json.dumps(next(self.responses))


def transcript(*, notes_final="fixture validated"):
    artifact = "artifact-" + hashlib.sha256(live._BUILD_LOG.encode()).hexdigest()
    bounds = len(live._DIAGNOSTIC.encode())
    evidence = (
        {"tool": "read_file", "path": "build.log"},
        {"tool": "read_file", "path": f".tool-artifacts/{artifact}/bytes/0-{bounds}"},
    )
    return (
        *live._BASE_ACTIONS,
        _VALIDATE,
        *evidence,
        {"final": "fixture validated"},
        _WRITE,
        _VALIDATE,
        *evidence,
        {"final": notes_final},
    )


def records(output, label):
    return [
        json.loads(line.removeprefix(f"{label}> "))
        for line in output.splitlines()
        if line.startswith(f"{label}> ")
    ]


def run_fixture(monkeypatch, tmp_path, answers, responses):
    answers = iter(answers)
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    monkeypatch.setattr(
        live,
        "TemporaryDirectory",
        lambda **options: TemporaryDirectory(dir=tmp_path, **options),
    )
    monkeypatch.setattr(
        "subprocess.run", lambda *args, **kwargs: pytest.fail("shell used")
    )
    with mx.stream(mx.cpu):
        live.run_live_fixture(api, TinyModel(), Tokenizer(responses))


@pytest.mark.parametrize("selected", ["validate-only", "add-notes"])
def test_live_fixture_pauses_and_selects_a_passing_branch(
    monkeypatch, tmp_path, capsys, selected
):
    run_fixture(monkeypatch, tmp_path, ["resume", selected], transcript())
    output = capsys.readouterr().out
    (pause,) = records(output, "pause")
    assert pause["last_action"] == {"tool": "read_file", "path": "app.py"}
    assert "Welcome" in pause["last_observation"]
    assert pause["denied_unapproved_write"]
    base = pause["receipts"]
    assert [receipt["tool"] for receipt in base] == ["edit_file", "run_command"]
    compacted, validate_evidence, notes_evidence = records(output, "context")
    assert compacted["tokens_before"] == pause["checkpoint_tokens"]
    tokens = compacted["tokens_after"]
    assert 0 < tokens < compacted["tokens_before"]
    assert compacted["receipt_ids"] == [base[1]["id"]]
    for evidence in (validate_evidence, notes_evidence):
        assert (
            evidence["range_bytes"]
            < evidence["inline_bytes"]
            <= 512
            < evidence["artifact_bytes"]
        )
    validate, notes = records(output, "branch")
    for branch in (validate, notes):
        assert branch["passed"] and branch["receipts"][:2] == base
        assert branch["reused_tokens"] == branch["cache"]["offset"] == tokens
        assert branch["cache"]["layer_offsets"] == [tokens]
    assert validate["name"] == "validate-only" and validate["changed"] == []
    assert notes["name"] == "add-notes" and notes["changed"] == ["NOTES.md"]
    assert [receipt["tool"] for receipt in notes["receipts"][2:]] == [
        "write_file",
        "run_command",
    ]
    assert records(output, "selected") == [{"name": selected, "passed": True}]
    assert not list(tmp_path.iterdir())


def test_live_fixture_refuses_a_failing_branch(monkeypatch, tmp_path, capsys):
    with pytest.raises(api.AgentError, match="selected branch must pass evaluation"):
        run_fixture(
            monkeypatch,
            tmp_path,
            ["resume", "add-notes"],
            transcript(notes_final="unfinished"),
        )
    output = capsys.readouterr().out
    assert [branch["passed"] for branch in records(output, "branch")] == [True, False]
    assert not records(output, "selected")
    assert not list(tmp_path.iterdir())


def test_live_fixture_stops_at_pause_without_resume(monkeypatch, tmp_path, capsys):
    with pytest.raises(api.AgentError, match="type resume"):
        run_fixture(monkeypatch, tmp_path, ["exit"], live._BASE_ACTIONS)
    output = capsys.readouterr().out
    assert records(output, "pause") and not records(output, "branch")
    assert not list(tmp_path.iterdir())


def test_live_cli_rejects_workspace_override(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "run_live", lambda *args: pytest.fail("model loaded"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--live-fixture", "--root", str(tmp_path)])
    assert error.value.code == 2
    assert "--live-fixture takes no other arguments" in capsys.readouterr().err


def test_live_cli_requires_the_cached_4b_snapshot(monkeypatch, capsys):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def missing(model, *, local_files_only):
        assert model == "Qwen/Qwen3-4B-MLX-4bit" and local_files_only
        raise LocalEntryNotFoundError("local snapshot is absent")

    monkeypatch.setattr("huggingface_hub.snapshot_download", missing)
    with pytest.raises(SystemExit) as error:
        cli.main(["--live-fixture"])
    assert error.value.code == 1
    assert "missing from the local model cache" in capsys.readouterr().err
