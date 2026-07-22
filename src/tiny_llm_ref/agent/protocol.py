from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .workspace import Workspace


class AgentError(ValueError):
    """A recoverable error that can be returned to the model."""


@dataclass(frozen=True)
class FinalAction:
    """Week 4, Day 2: a model response that finishes the task."""

    final: str


@dataclass(frozen=True)
class ToolAction:
    """Week 4, Day 2: one validated tool request from the model."""

    tool: str
    arguments: dict[str, Any]


AgentAction = FinalAction | ToolAction


TOOL_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "list_files": (frozenset(), frozenset({"path"})),
    "read_file": (frozenset({"path"}), frozenset()),
    "write_file": (frozenset({"path", "content"}), frozenset()),
    "edit_file": (frozenset({"path", "old", "new"}), frozenset()),
    "run_command": (frozenset({"argv"}), frozenset()),
}


def parse_action(
    response: str,
    available_tools: frozenset[str] | None = None,
) -> AgentAction:
    """Week 4, Day 2: strictly parse and validate exactly one JSON action."""

    try:
        raw = json.loads(response)
    except json.JSONDecodeError as error:
        raise AgentError(f"response is not valid JSON: {error.msg}") from error
    if not isinstance(raw, dict):
        raise AgentError("response must be a JSON object")

    if "final" in raw:
        if set(raw) != {"final"} or not isinstance(raw["final"], str):
            raise AgentError("final action must contain only a string 'final' field")
        if not raw["final"].strip():
            raise AgentError("final response must not be empty")
        return FinalAction(raw["final"])

    tool = raw.get("tool")
    if not isinstance(tool, str):
        raise AgentError("tool action requires a string 'tool' field")
    if tool not in TOOL_FIELDS:
        raise AgentError(f"unknown tool: {tool}")
    if available_tools is not None and tool not in available_tools:
        raise AgentError(f"tool is not enabled: {tool}")

    required, optional = TOOL_FIELDS[tool]
    supplied = set(raw) - {"tool"}
    missing = required - supplied
    extra = supplied - required - optional
    if missing:
        raise AgentError(f"missing fields for {tool}: {', '.join(sorted(missing))}")
    if extra:
        raise AgentError(f"unexpected fields for {tool}: {', '.join(sorted(extra))}")

    arguments = {name: raw[name] for name in supplied}
    if "path" in arguments and not isinstance(arguments["path"], str):
        raise AgentError("path must be a string")
    for name in ("content", "old", "new"):
        if name in arguments and not isinstance(arguments[name], str):
            raise AgentError(f"{name} must be a string")
    if "argv" in arguments and (
        not isinstance(arguments["argv"], list)
        or not arguments["argv"]
        or any(not isinstance(part, str) or not part for part in arguments["argv"])
    ):
        raise AgentError("argv must be a non-empty array of non-empty strings")
    return ToolAction(tool, arguments)


def build_system_prompt(workspace: Workspace) -> str:
    """Week 4, Day 2: describe only the tools authorized for this run."""

    lines = [
        "You are a coding agent. Inspect the workspace before editing it.",
        "Reply with exactly one JSON object and no markdown.",
        'Finish with: {"final":"brief summary"}',
        "Available actions:",
        '{"tool":"list_files","path":"."}',
        '{"tool":"read_file","path":"README.md"}',
    ]
    if workspace.policy.allow_writes:
        lines += [
            '{"tool":"write_file","path":"hello.py","content":"..."}',
            '{"tool":"edit_file","path":"hello.py","old":"...","new":"..."}',
            "Read an existing file before changing it.",
        ]
    else:
        lines.append("This run is read-only. Do not request file changes.")
    if workspace.policy.allowed_commands:
        lines.append("The operator allowed only these exact command arrays:")
        lines += [
            json.dumps(list(command))
            for command in workspace.policy.allowed_commands
        ]
        lines.append('Use: {"tool":"run_command","argv":["exact","arguments"]}')
    else:
        lines.append("Command execution is disabled.")
    lines += [
        "Paths must be relative to the workspace.",
        "Keep changes small and never invent file contents.",
    ]
    return "\n".join(lines)
