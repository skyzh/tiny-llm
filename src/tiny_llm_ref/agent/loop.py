import json
from collections.abc import Callable
from dataclasses import dataclass

from .context import append_tool_result, compact_messages
from .generation import Generate, initial_messages
from .protocol import (
    AgentAction,
    AgentError,
    FinalAction,
    build_system_prompt,
    parse_action,
)
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

        if min(
            self.max_steps,
            self.max_context_chars,
            self.max_invalid_actions,
            self.max_identical_actions,
        ) <= 0:
            raise ValueError("agent limits must be positive")


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

    limits = limits or AgentLimits()
    messages = initial_messages(task, build_system_prompt(workspace))
    events: list[AgentEvent] = []
    invalid_actions = 0
    previous_signature: str | None = None
    identical_actions = 0

    def finish(completed: bool, reason: str, final: str | None = None) -> AgentRun:
        """Week 4, Day 7: freeze the trace and modified paths into a result."""

        modified = tuple(
            sorted(
                str(path.relative_to(workspace.policy.root))
                for path in workspace.modified_files
            )
        )
        return AgentRun(completed, reason, final, tuple(events), modified)

    initial_chars = sum(len(message["content"]) for message in messages)
    if initial_chars > limits.max_context_chars:
        return finish(False, "context_limit")

    for step in range(1, limits.max_steps + 1):
        response = generate(messages)
        try:
            action = parse_action(response, workspace.available_tools)
        except AgentError as error:
            invalid_actions += 1
            previous_signature = None
            identical_actions = 0
            result = f"error: {error}"
            event = AgentEvent(step, response, None, result)
            events.append(event)
            if on_event is not None:
                on_event(event)
            if invalid_actions >= limits.max_invalid_actions:
                return finish(False, "invalid_action_limit")
            messages = append_tool_result(messages, response, result)
            messages = compact_messages(messages, limits.max_context_chars)
            continue

        if isinstance(action, FinalAction):
            event = AgentEvent(step, response, action, None)
            events.append(event)
            if on_event is not None:
                on_event(event)
            return finish(True, "completed", action.final)

        signature = json.dumps(
            {"tool": action.tool, **action.arguments}, sort_keys=True
        )
        if signature == previous_signature:
            identical_actions += 1
        else:
            previous_signature = signature
            identical_actions = 1
        if identical_actions > limits.max_identical_actions:
            result = "error: identical action loop detected"
            event = AgentEvent(step, response, action, result)
            events.append(event)
            if on_event is not None:
                on_event(event)
            return finish(False, "repeated_action_limit")

        result = workspace.execute(action)
        event = AgentEvent(step, response, action, result)
        events.append(event)
        if on_event is not None:
            on_event(event)
        messages = append_tool_result(messages, response, result)
        messages = compact_messages(messages, limits.max_context_chars)

    return finish(False, "step_limit")
