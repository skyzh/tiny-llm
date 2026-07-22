from collections.abc import Callable
from dataclasses import dataclass

from .generation import Generate
from .protocol import AgentAction
from .workspace import Workspace


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


def run_agent(
    task: str,
    generate: Generate,
    workspace: Workspace,
    limits: AgentLimits | None = None,
    on_event: Callable[[AgentEvent], None] | None = None,
) -> AgentRun:
    """Week 4, Day 6: run the bounded observe-act loop with recovery and tracing."""

    pass
