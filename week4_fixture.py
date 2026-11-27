"""A fixed, disposable smoke test for the Week 4 parser and dispatcher."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
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
_EXPECTED_FILES = (("app.py", _EXPECTED_APP), ("NOTES.md", _NOTES.encode()))


def _fixture_workspace(
    api,
    root: Path,
    store,
    *,
    approve=True,
    allow_effects=True,
    paths=_PATHS,
    allowed_effects=(_EDIT, _WRITE, _VALIDATE),
    expected_files=_EXPECTED_FILES,
    detailed_validation=False,
):
    paths = paths if allow_effects else frozenset({"app.py"})

    class FixtureWorkspace(api.Workspace):
        @property
        def available_tools(self):
            return super().available_tools & _TOOLS

        def resolve_path(self, raw, *, must_exist=True):
            path = super().resolve_path(raw, must_exist=must_exist)
            if raw != "." and raw not in paths:
                raise api.AgentError("path is outside the fixture allowlist")
            return path

        def list_files(self, raw="."):
            listing = super().list_files(raw)
            visible = [
                line
                for line in listing.splitlines()
                if line.removeprefix("file ") in paths and line.startswith("file ")
            ]
            return "\n".join(visible) or "(empty directory)"

        def run_command(self, argv):
            self._allowed_command(argv)
            report = ["status: 0", "output:", "fixture bytes match"]
            for name, expected in expected_files:
                try:
                    path = self.resolve_path(name)
                    with path.open("rb") as source:
                        observed = source.read(len(expected) + 1)
                except (OSError, api.AgentError):
                    return "status: 1\noutput:\nfixture bytes differ"
                if observed != expected:
                    return "status: 1\noutput:\nfixture bytes differ"
                if detailed_validation:
                    report.append(
                        f"{name} matched all {len(observed)} bytes.\n"
                        f"Observed SHA-256 {hashlib.sha256(observed).hexdigest()}\n"
                        f"Expected SHA-256 {hashlib.sha256(expected).hexdigest()}"
                    )
            return "\n".join(report)

    def confirm(action):
        return approve and {"tool": action.tool, **action.arguments} in allowed_effects

    return FixtureWorkspace(
        api.ToolPolicy(
            root,
            allow_writes=allow_effects,
            allowed_commands=(_VALIDATION,) if allow_effects else (),
        ),
        confirm,
        store,
    )


def inspection_backend():
    from tiny_llm_ref import agent, models

    return agent, models


@contextmanager
def inspection_workspace(api):
    """Give the cached model read-only access to the supplied greeting."""
    with TemporaryDirectory(prefix="tiny-llm-week4-inspect-") as temporary:
        root = Path(temporary)
        (root / "app.py").write_bytes(_APP)
        yield _fixture_workspace(api, root, api.ReceiptStore(), allow_effects=False)


def inspection_messages(messages):
    file_result = {"role": "user", "content": "Tool result:\n" + _APP.decode()}
    if file_result in messages[2:]:
        instruction = (
            "The file has been read. Use the tool result to describe the greeting. "
            'Return one JSON object with only a "final" string.'
        )
    else:
        instruction = (
            "Request the file by replying with exactly this JSON: "
            '{"tool":"read_file","path":"app.py"}'
        )
    return [{"role": "system", "content": instruction}, *messages[1:]]


def observed_fixture_read(api, run):
    return any(
        isinstance(event.action, api.ToolAction)
        and event.action.tool == "read_file"
        and event.action.arguments == {"path": "app.py"}
        and event.result == _APP.decode()
        for event in run.events
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
