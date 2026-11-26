"""A fixed, disposable smoke test for the Week 4 parser and dispatcher."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

_APP = b'def greeting(name):\n    return f"Hello, {name}!"\n'
_EXPECTED_APP = _APP.replace(b"Hello", b"Welcome")
_NOTES = "Greeting now says Welcome.\n"
_VALIDATION = ("validate_fixture",)
_PATHS = frozenset({"app.py", "NOTES.md"})
_TOOLS = frozenset(
    {"list_files", "read_file", "edit_file", "write_file", "run_command"}
)
_EDIT = {"tool": "edit_file", "path": "app.py", "old": "Hello", "new": "Welcome"}
_WRITE = {"tool": "write_file", "path": "NOTES.md", "content": _NOTES}
_VALIDATE = {"tool": "run_command", "argv": list(_VALIDATION)}


def _fixture_workspace(api, root: Path, store, *, approve=True):
    class FixtureWorkspace(api.Workspace):
        @property
        def available_tools(self):
            return super().available_tools & _TOOLS

        def resolve_path(self, raw, *, must_exist=True):
            path = super().resolve_path(raw, must_exist=must_exist)
            if raw != "." and raw not in _PATHS:
                raise api.AgentError("path is outside the fixture allowlist")
            return path

        def list_files(self, raw="."):
            listing = super().list_files(raw)
            visible = [
                line
                for line in listing.splitlines()
                if line.removeprefix("file ") in _PATHS and line.startswith("file ")
            ]
            return "\n".join(visible) or "(empty directory)"

        def run_command(self, argv):
            self._allowed_command(argv)
            for name, expected in (
                ("app.py", _EXPECTED_APP),
                ("NOTES.md", _NOTES.encode()),
            ):
                try:
                    path = self.resolve_path(name)
                    with path.open("rb") as source:
                        matched = source.read(len(expected) + 1) == expected
                except (OSError, api.AgentError):
                    matched = False
                if not matched:
                    return "status: 1\noutput:\nfixture bytes differ"
            return "status: 0\noutput:\nfixture bytes match"

    def confirm(action):
        return approve and {"tool": action.tool, **action.arguments} in (
            _EDIT,
            _WRITE,
            _VALIDATE,
        )

    return FixtureWorkspace(
        api.ToolPolicy(root, allow_writes=True, allowed_commands=(_VALIDATION,)),
        confirm,
        store,
    )


def run_fixture():
    from tiny_llm_ref import agent as api

    with TemporaryDirectory(prefix="tiny-llm-week4-fixture-") as temporary:
        root = Path(temporary)
        workspace_root = root / "workspace"
        workspace_root.mkdir()
        ledger = root / "receipts" / "effects.jsonl"
        ledger.parent.mkdir()
        app = workspace_root / "app.py"
        app.write_bytes(_APP)
        store = api.ReceiptStore(ledger)
        workspace = _fixture_workspace(api, workspace_root, store)
        actions = (
            {"tool": "list_files"},
            {"tool": "read_file", "path": "app.py"},
            {**_WRITE, "content": "Unapproved notes.\n"},
            _EDIT,
            _WRITE,
            _VALIDATE,
            {"final": "Greeting and notes match the fixture."},
        )
        responses = iter(map(json.dumps, actions))
        run = api.run_agent(
            "Inspect the fixture, update its greeting and notes, and validate its bytes.",
            lambda _messages: next(responses),
            workspace,
        )
        records = [
            api.EffectReceipt.from_dict(json.loads(line))
            for line in ledger.read_text(encoding="utf-8").splitlines()
        ]
        if (
            not run.completed
            or run.events[0].result != "file app.py"
            or run.events[1].result != _APP.decode()
            or run.events[2].result != "error: operator denied the tool action"
            or run.events[5].result != "status: 0\noutput:\nfixture bytes match"
            or app.read_bytes() != _EXPECTED_APP
            or (workspace_root / "NOTES.md").read_bytes() != _NOTES.encode()
            or workspace.modified_files != ("NOTES.md", "app.py")
            or [receipt.tool for receipt in records]
            != ["edit_file", "write_file", "run_command"]
            or any(receipt.exit_state != "ok" for receipt in records)
        ):
            raise RuntimeError(
                "the fixture did not produce the expected observations and receipts"
            )
        return {
            "listed": run.events[0].result,
            "read": run.events[1].result,
            "denied": run.events[2].result,
            "changed": list(workspace.modified_files),
            "validation": run.events[5].result,
            "receipts": [receipt.tool for receipt in records],
        }
