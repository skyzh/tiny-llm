"""The supplied CLI fixture and receipt boundary use the real Week 4 tools."""

import json
import os
from pathlib import Path

import pytest

import agent as cli
from agent_cli_receipts import manual_workspace
from tiny_llm_ref import agent as api
from week4_fixture import _APP, _EDIT, _NOTES, _VALIDATE, _WRITE, _fixture_workspace


def _call(workspace, payload):
    action = api.parse_action(json.dumps(payload), workspace.available_tools)
    return workspace.execute(action)


def test_cli_fixed_fixture_uses_no_subprocess(monkeypatch, capsys):
    def fail(*args, **kwargs):
        pytest.fail("the fixture must validate in process")

    monkeypatch.setattr("subprocess.run", fail)
    assert cli.main(["--autoreject-fixture"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record == {
        "listed": "file app.py",
        "read": _APP.decode(),
        "denied": "error: operator denied the tool action",
        "changed": ["NOTES.md", "app.py"],
        "validation": "status: 0\noutput:\nfixture bytes match",
        "receipts": ["edit_file", "write_file", "run_command"],
    }


def test_cli_fixture_rejects_user_workspace(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--autoreject-fixture", "--root", str(tmp_path)])
    assert error.value.code == 2
    assert "--autoreject-fixture takes no other arguments" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_fixture_dispatch_denial_leaves_files_and_ledger_unchanged(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "app.py").write_bytes(_APP)
    external = tmp_path / "inert.txt"
    external.write_bytes(b"fixture sentinel\n")
    (root / "leaf").symlink_to(external)
    (root / "alias").symlink_to(tmp_path, target_is_directory=True)
    ledger = tmp_path / "receipts" / "effects.jsonl"
    ledger.parent.mkdir()
    store = api.ReceiptStore(ledger)
    workspace = _fixture_workspace(api, root, store)
    unapproved = _fixture_workspace(api, root, store, approve=False)

    assert _call(workspace, {"tool": "read_file", "path": "app.py"}) == _APP.decode()
    assert _call(unapproved, {"tool": "read_file", "path": "app.py"}) == _APP.decode()
    assert "denied" in _call(unapproved, _EDIT)
    assert not ledger.exists()
    assert _call(workspace, _EDIT).startswith("edited ")
    assert _call(workspace, _WRITE).startswith("wrote ")
    assert _call(workspace, _VALIDATE) == "status: 0\noutput:\nfixture bytes match"

    paths = (root / "app.py", root / "NOTES.md", ledger, external)
    before = tuple(path.read_bytes() for path in paths)
    requests = [
        {**_WRITE, "path": str(external)},
        {**_WRITE, "path": "../inert.txt"},
        {**_WRITE, "path": ".env"},
        {**_WRITE, "path": "leaf"},
        {**_WRITE, "path": "alias/inert.txt"},
        {**_WRITE, "path": "../receipts/effects.jsonl"},
        {**_WRITE, "path": "other.txt"},
        {**_EDIT, "old": "Welcome", "new": "Another greeting"},
        {**_VALIDATE, "argv": ["unavailable_check"]},
    ]
    assert (
        _call(workspace, {"tool": "read_file", "path": "app.py"}).count("Welcome") == 1
    )
    for payload in requests:
        assert _call(workspace, payload).startswith("error:")
        assert tuple(path.read_bytes() for path in paths) == before
        assert not (root / "other.txt").exists()
    with pytest.raises(api.AgentError, match="unknown tool"):
        _call(workspace, {"tool": "remove_file", "path": "app.py"})
    assert tuple(path.read_bytes() for path in paths) == before


def test_fixture_validator_checks_contents_without_running_them(tmp_path):
    (tmp_path / "app.py").write_bytes(_APP)
    (tmp_path / "NOTES.md").write_text(_NOTES, encoding="utf-8")
    store = api.ReceiptStore()
    workspace = _fixture_workspace(api, tmp_path, store)
    assert _call(workspace, _VALIDATE) == "status: 1\noutput:\nfixture bytes differ"
    assert store.get("call-1").exit_state == "error"


def test_manual_cli_rejects_linked_and_nonregular_receipt_paths(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    target = outside / "inert.jsonl"
    target.write_bytes(b"")
    (root / "linked-file").symlink_to(target)
    (root / "linked-dir").symlink_to(outside, target_is_directory=True)
    (root / "directory-log").mkdir()
    os.link(target, root / "shared-file")
    policy = api.ToolPolicy(root)
    for path in (
        "linked-file",
        "linked-dir/new.jsonl",
        "directory-log",
        "shared-file",
        str(target),
        "../new.jsonl",
        "missing/new.jsonl",
        ".git/effects.jsonl",
    ):
        with pytest.raises(ValueError, match="--receipt-log"):
            manual_workspace(api, policy, Path(path), lambda _action: True)
    assert list(outside.iterdir()) == [target]
    assert target.read_bytes() == b""


def test_manual_cli_keeps_the_log_out_of_file_tools(tmp_path):
    (tmp_path / "app.py").write_bytes(_APP)
    log_name = Path("effects.jsonl")
    policy = api.ToolPolicy(tmp_path, allow_writes=True)
    workspace, ledger = manual_workspace(api, policy, log_name, lambda _action: True)
    for name in ("effects.jsonl", "EFFECTS.JSONL"):
        assert _call(workspace, {**_WRITE, "path": name}).startswith("error:")
    assert not ledger.exists()
    assert _call(workspace, _WRITE).startswith("wrote ")
    before = ledger.read_bytes()
    os.link(ledger, tmp_path / "alias.jsonl")
    assert _call(workspace, {"tool": "list_files"}) == "file NOTES.md\nfile app.py"
    for name in (str(log_name), "alias.jsonl"):
        for payload in (
            {"tool": "read_file", "path": name},
            {**_WRITE, "path": name},
            {**_EDIT, "path": name},
        ):
            assert _call(workspace, payload).startswith("error:")
    assert ledger.read_bytes() == before
    (tmp_path / "alias.jsonl").unlink()
    restarted, _ = manual_workspace(api, policy, log_name, lambda _action: True)
    assert restarted.receipt_store.get("call-3").tool == "write_file"
