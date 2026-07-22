from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


__all__ = [
    "AgentError",
    "AgentEvent",
    "AgentLimits",
    "AgentRun",
    "FinalAction",
    "ToolAction",
    "ToolPolicy",
    "Workspace",
    "append_tool_result",
    "build_system_prompt",
    "compact_messages",
    "generate_response",
    "initial_messages",
    "parse_action",
    "run_agent",
]


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
Message = dict[str, str]
Generate = Callable[[list[Message]], str]


@dataclass(frozen=True)
class ToolPolicy:
    """Week 4, Day 3: filesystem and command boundaries for one workspace."""

    root: Path
    allow_writes: bool = False
    allowed_commands: tuple[tuple[str, ...], ...] = ()
    max_file_bytes: int = 64 * 1024
    max_write_bytes: int = 64 * 1024
    max_list_entries: int = 200
    max_tool_output_chars: int = 16_000
    command_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Week 4, Day 3: normalize the root and reject invalid limits."""

        pass


@dataclass(frozen=True)
class AgentLimits:
    """Week 4, Day 6: budgets that guarantee the loop eventually stops."""

    max_steps: int = 8
    max_context_chars: int = 48_000
    max_invalid_actions: int = 3
    max_identical_actions: int = 2

    def __post_init__(self) -> None:
        """Week 4, Day 6: reject budgets that could disable a stop condition."""

        pass


@dataclass(frozen=True)
class AgentEvent:
    """Week 4, Day 7: one auditable model/action/tool interaction."""

    step: int
    response: str
    action: AgentAction | None
    result: str | None


@dataclass(frozen=True)
class AgentRun:
    """Week 4, Day 7: the complete, measurable result of an agent task."""

    completed: bool
    reason: str
    final: str | None
    events: tuple[AgentEvent, ...]
    modified_files: tuple[str, ...]


def initial_messages(task: str, system_prompt: str) -> list[Message]:
    """Week 4, Day 1: create the durable beginning of an agent conversation."""

    pass


def generate_response(
    model,
    tokenizer,
    messages: list[Message],
    cache_factory: Callable[[], Sequence[Any]],
    max_tokens: int,
    enable_thinking: bool = False,
) -> str:
    """Week 4, Day 1: decode one action with the course model and a fresh cache."""

    pass


def parse_action(
    response: str,
    available_tools: frozenset[str] | None = None,
) -> AgentAction:
    """Week 4, Day 2: strictly parse and validate exactly one JSON action."""

    pass


def compact_messages(messages: list[Message], max_chars: int) -> list[Message]:
    """Week 4, Day 4: retain task anchors and the newest complete tool turns."""

    pass


@dataclass
class Workspace:
    """Week 4, Days 3 and 5: bounded tools over one explicit workspace root."""

    policy: ToolPolicy
    observed_files: set[Path] = field(default_factory=set, init=False)
    modified_files: set[Path] = field(default_factory=set, init=False)

    @property
    def available_tools(self) -> frozenset[str]:
        """Week 4, Day 3: expose only tools enabled by operator policy."""

        pass

    def resolve_path(self, raw: str, *, must_exist: bool = True) -> Path:
        """Week 4, Day 3: reject escapes, protected files, and symlink traversal."""

        pass

    def _reject_protected(self, relative: Path) -> None:
        """Week 4, Day 3: keep repository metadata and common secrets hidden."""

        pass

    def list_files(self, raw: str = ".") -> str:
        """Week 4, Day 3: list one directory without following symlinks."""

        pass

    def read_file(self, raw: str) -> str:
        """Week 4, Day 3: read one bounded UTF-8 regular file."""

        pass

    def write_file(self, raw: str, content: str) -> str:
        """Week 4, Day 3: atomically create or replace an inspected file."""

        pass

    def edit_file(self, raw: str, old: str, new: str) -> str:
        """Week 4, Day 5: make one exact, reviewable replacement in a read file."""

        pass

    def run_command(self, argv: list[str]) -> str:
        """Week 4, Day 5: run only an operator-approved exact argv without a shell."""

        pass

    def execute(self, action: ToolAction) -> str:
        """Week 4, Day 3: dispatch a validated action and return recoverable errors."""

        pass

    def _atomic_write(self, path: Path, content: bytes) -> None:
        """Week 4, Day 3: replace a file without exposing a partial write."""

        pass


def build_system_prompt(workspace: Workspace) -> str:
    """Week 4, Day 2: describe only the tools authorized for this run."""

    pass


def append_tool_result(
    messages: list[Message], response: str, result: str
) -> list[Message]:
    """Week 4, Day 4: record an action and its observation as one complete turn."""

    pass


def run_agent(
    task: str,
    generate: Generate,
    workspace: Workspace,
    limits: AgentLimits | None = None,
    on_event: Callable[[AgentEvent], None] | None = None,
) -> AgentRun:
    """Week 4, Day 6: run the bounded observe-act loop with recovery and tracing."""

    pass
