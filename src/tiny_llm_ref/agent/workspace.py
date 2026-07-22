import heapq
import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread

from .protocol import AgentError, ToolAction


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
