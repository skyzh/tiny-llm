from dataclasses import dataclass, field
from pathlib import Path

from .protocol import ToolAction


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
