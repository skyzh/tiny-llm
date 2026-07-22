from __future__ import annotations

import heapq
import json
import os
import signal
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
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

        object.__setattr__(self, "root", self.root.resolve())
        numeric_limits = (
            self.max_file_bytes,
            self.max_write_bytes,
            self.max_list_entries,
            self.max_tool_output_chars,
            self.command_timeout_seconds,
        )
        if any(limit <= 0 for limit in numeric_limits):
            raise ValueError("tool policy limits must be positive")
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError("workspace root must be an existing directory")
        for command in self.allowed_commands:
            if not command or any(
                not isinstance(part, str) or not part or "\x00" in part
                for part in command
            ):
                raise ValueError("allowed commands must contain non-empty arguments")


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


TOOL_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "list_files": (frozenset(), frozenset({"path"})),
    "read_file": (frozenset({"path"}), frozenset()),
    "write_file": (frozenset({"path", "content"}), frozenset()),
    "edit_file": (frozenset({"path", "old", "new"}), frozenset()),
    "run_command": (frozenset({"argv"}), frozenset()),
}


def initial_messages(task: str, system_prompt: str) -> list[Message]:
    """Week 4, Day 1: create the durable beginning of an agent conversation."""

    if not task.strip():
        raise ValueError("task must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]


def generate_response(
    model,
    tokenizer,
    messages: list[Message],
    cache_factory: Callable[[], Sequence[Any]],
    max_tokens: int,
    enable_thinking: bool = False,
) -> str:
    """Week 4, Day 1: decode one action with the course model and a fresh cache."""

    import mlx.core as mx

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    caches = list(cache_factory())
    output: list[int] = []
    offset = 0
    try:
        for _ in range(max_tokens):
            logits = model(tokens[None], offset, caches)[:, -1, :]
            token = int(mx.argmax(logits, axis=-1).item())
            if token == tokenizer.eos_token_id:
                break
            output.append(token)
            offset += tokens.size
            tokens = mx.array([token])
        return tokenizer.decode(output)
    finally:
        for cache in caches:
            release = getattr(cache, "release", None)
            if release is not None:
                release()


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


def compact_messages(messages: list[Message], max_chars: int) -> list[Message]:
    """Week 4, Day 4: retain task anchors and the newest complete tool turns."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    total_chars = sum(len(item["content"]) for item in messages)
    if len(messages) <= 2 or total_chars <= max_chars:
        return list(messages)

    anchors = list(messages[:2])
    anchor_chars = sum(len(item["content"]) for item in anchors)
    if anchor_chars >= max_chars:
        return anchors

    turns = [messages[index : index + 2] for index in range(2, len(messages), 2)]
    kept: list[list[Message]] = []
    used = anchor_chars
    for turn in reversed(turns):
        turn_chars = sum(len(item["content"]) for item in turn)
        if used + turn_chars > max_chars:
            break
        kept.append(turn)
        used += turn_chars
    return anchors + [item for turn in reversed(kept) for item in turn]


@dataclass
class Workspace:
    """Week 4, Days 3 and 5: bounded tools over one explicit workspace root."""

    policy: ToolPolicy
    observed_files: set[Path] = field(default_factory=set, init=False)
    modified_files: set[Path] = field(default_factory=set, init=False)

    @property
    def available_tools(self) -> frozenset[str]:
        """Week 4, Day 3: expose only tools enabled by operator policy."""

        tools = {"list_files", "read_file"}
        if self.policy.allow_writes:
            tools.update({"write_file", "edit_file"})
        if self.policy.allowed_commands:
            tools.add("run_command")
        return frozenset(tools)

    def resolve_path(self, raw: str, *, must_exist: bool = True) -> Path:
        """Week 4, Day 3: reject escapes, protected files, and symlink traversal."""

        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise AgentError("path must be a non-empty string")
        unresolved = Path(raw)
        if unresolved.is_absolute():
            raise AgentError("absolute paths are not allowed")
        if ".." in unresolved.parts:
            raise AgentError("parent path components are not allowed")
        current = self.policy.root
        for part in unresolved.parts:
            current /= part
            if current.is_symlink():
                raise AgentError("symlinks are not accessible")
        path = (self.policy.root / unresolved).resolve(strict=False)
        try:
            relative = path.relative_to(self.policy.root)
        except ValueError as error:
            raise AgentError("path leaves the workspace") from error
        self._reject_protected(relative)

        if must_exist and not path.exists():
            raise AgentError(f"path does not exist: {relative}")
        return path

    def _reject_protected(self, relative: Path) -> None:
        """Week 4, Day 3: keep repository metadata and common secrets hidden."""

        secret_names = {
            ".env",
            ".aws",
            ".azure",
            ".docker",
            ".gnupg",
            ".kube",
            ".netrc",
            ".npmrc",
            ".pypirc",
            ".ssh",
            "id_rsa",
            "id_ed25519",
        }
        for part in relative.parts:
            lower = part.lower()
            if lower == ".git":
                raise AgentError(".git is not accessible")
            if lower in secret_names or lower.startswith(".env."):
                raise AgentError("potential secret files are not accessible")
            if lower.endswith((".pem", ".key")):
                raise AgentError("potential key files are not accessible")

    def list_files(self, raw: str = ".") -> str:
        """Week 4, Day 3: list one directory without following symlinks."""

        path = self.resolve_path(raw)
        if not path.is_dir():
            raise AgentError("list_files path must be a directory")
        lines: list[str] = []
        items = heapq.nsmallest(
            self.policy.max_list_entries,
            path.iterdir(),
            key=lambda entry: entry.name,
        )
        for item in items:
            if item.is_symlink():
                continue
            try:
                relative = item.relative_to(self.policy.root)
                self._reject_protected(relative)
            except AgentError:
                continue
            kind = "dir" if item.is_dir() else "file"
            lines.append(f"{kind} {relative}")
            if len(lines) == self.policy.max_list_entries:
                break
        return "\n".join(lines)

    def read_file(self, raw: str) -> str:
        """Week 4, Day 3: read one bounded UTF-8 regular file."""

        path = self.resolve_path(raw)
        if not path.is_file():
            raise AgentError("read_file path must be a regular file")
        with path.open("rb") as file:
            data = file.read(self.policy.max_file_bytes + 1)
        if len(data) > self.policy.max_file_bytes:
            raise AgentError(f"file exceeds {self.policy.max_file_bytes} bytes")
        content = data.decode("utf-8")
        self.observed_files.add(path)
        return content

    def write_file(self, raw: str, content: str) -> str:
        """Week 4, Day 3: atomically create or replace an inspected file."""

        if not self.policy.allow_writes:
            raise AgentError("writes are disabled; restart with --allow-writes")
        encoded = content.encode("utf-8")
        if len(encoded) > self.policy.max_write_bytes:
            raise AgentError(f"content exceeds {self.policy.max_write_bytes} bytes")
        path = self.resolve_path(raw, must_exist=False)
        if path.exists() and path not in self.observed_files:
            raise AgentError("existing files must be read before they are overwritten")
        if path.exists() and not path.is_file():
            raise AgentError("write_file path must be a regular file")
        if not path.parent.exists() or not path.parent.is_dir():
            raise AgentError("parent directory must already exist")
        self._atomic_write(path, encoded)
        self.observed_files.add(path)
        self.modified_files.add(path)
        return f"wrote {path.relative_to(self.policy.root)}"

    def edit_file(self, raw: str, old: str, new: str) -> str:
        """Week 4, Day 5: make one exact, reviewable replacement in a read file."""

        path = self.resolve_path(raw)
        if path not in self.observed_files:
            raise AgentError("files must be read before they are edited")
        if not old:
            raise AgentError("old text must not be empty")
        content = self.read_file(raw)
        matches = content.count(old)
        if matches != 1:
            raise AgentError(f"old text must match exactly once; found {matches}")
        return self.write_file(raw, content.replace(old, new, 1))

    def run_command(self, argv: list[str]) -> str:
        """Week 4, Day 5: run only an operator-approved exact argv without a shell."""

        command = tuple(argv)
        if command not in self.policy.allowed_commands:
            raise AgentError("command was not explicitly allowed by the operator")
        environment = {
            name: value
            for name in ("PATH", "LANG", "LC_ALL", "TMPDIR")
            if (value := os.environ.get(name)) is not None
        }
        process = subprocess.Popen(
            argv,
            cwd=self.policy.root,
            env=environment,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        captured = bytearray()
        output_truncated = Event()

        def drain_output() -> None:
            """Week 4, Day 5: drain output without retaining unbounded bytes."""

            try:
                while chunk := process.stdout.read(4096):
                    remaining = self.policy.max_tool_output_chars - len(captured)
                    if remaining > 0:
                        captured.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        output_truncated.set()
            except (OSError, ValueError):
                pass

        reader = Thread(target=drain_output, daemon=True)
        reader.start()
        try:
            returncode = process.wait(timeout=self.policy.command_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, AttributeError):
                process.kill()
            process.wait()
            raise AgentError(
                f"command exceeded {self.policy.command_timeout_seconds:g} seconds"
            ) from error
        finally:
            reader.join(timeout=1)
            if reader.is_alive():
                process.stdout.close()
                reader.join(timeout=1)

        output = captured.decode("utf-8", errors="replace")
        if output_truncated.is_set():
            output += "\n... command output truncated"
        return f"exit code: {returncode}\n{output}".rstrip()

    def execute(self, action: ToolAction) -> str:
        """Week 4, Day 3: dispatch a validated action and return recoverable errors."""

        try:
            if action.tool == "list_files":
                result = self.list_files(action.arguments.get("path", "."))
            elif action.tool == "read_file":
                result = self.read_file(action.arguments["path"])
            elif action.tool == "write_file":
                result = self.write_file(
                    action.arguments["path"], action.arguments["content"]
                )
            elif action.tool == "edit_file":
                result = self.edit_file(
                    action.arguments["path"],
                    action.arguments["old"],
                    action.arguments["new"],
                )
            elif action.tool == "run_command":
                result = self.run_command(action.arguments["argv"])
            else:
                raise AgentError(f"unknown tool: {action.tool}")
        except (
            KeyError,
            OSError,
            subprocess.SubprocessError,
            ValueError,
        ) as error:
            result = f"error: {error}"
        marker = "\n... tool output truncated"
        if len(result) > self.policy.max_tool_output_chars:
            if self.policy.max_tool_output_chars <= len(marker):
                return result[: self.policy.max_tool_output_chars]
            result = result[: self.policy.max_tool_output_chars - len(marker)] + marker
        return result

    def _atomic_write(self, path: Path, content: bytes) -> None:
        """Week 4, Day 3: replace a file without exposing a partial write."""

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".tiny-llm-agent-", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            if path.exists():
                os.chmod(temporary, path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


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


def append_tool_result(
    messages: list[Message], response: str, result: str
) -> list[Message]:
    """Week 4, Day 4: record an action and its observation as one complete turn."""

    return messages + [
        {"role": "assistant", "content": response},
        {"role": "user", "content": f"Tool result:\n{result}"},
    ]


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
