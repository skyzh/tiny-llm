"""Receipt storage for the interactive Week 4 CLI."""

from __future__ import annotations

import json
import os
import stat
import unicodedata
from contextlib import ExitStack
from pathlib import Path


def _open_log(root: Path, relative: Path, *, append: bool):
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_NOFOLLOW | os.O_NONBLOCK
    file_flags |= os.O_WRONLY | os.O_APPEND | os.O_CREAT if append else os.O_RDONLY
    with ExitStack() as stack:
        directory = os.open(root, directory_flags)
        stack.callback(os.close, directory)
        for part in relative.parts[:-1]:
            directory = os.open(part, directory_flags, dir_fd=directory)
            stack.callback(os.close, directory)
        try:
            fd = os.open(relative.name, file_flags, 0o600, dir_fd=directory)
        except FileNotFoundError:
            if not append:
                return None
            raise
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("--receipt-log must be a regular file with one link")
            return os.fdopen(fd, "a" if append else "r", encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise


def manual_workspace(api, policy, raw: Path | None, confirm):
    """Reserve the CLI log and retain the learner workspace's dispatch flow."""
    if raw is None:
        return api.Workspace(policy, confirm), None
    if (
        not raw.name
        or raw.is_absolute()
        or ".." in raw.parts
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in str(raw))
    ):
        raise ValueError("--receipt-log must be a workspace-relative file path")
    try:
        log_path = api.Workspace(policy).resolve_path(str(raw), must_exist=False)
    except api.AgentError as error:
        raise ValueError("--receipt-log path is inaccessible") from error

    class CliReceiptStore(api.ReceiptStore):
        def __init__(self):
            super().__init__()
            try:
                source = _open_log(policy.root, raw, append=False)
                if source is not None:
                    with source:
                        for line in source:
                            super().put(api.EffectReceipt.from_dict(json.loads(line)))
            except (OSError, ValueError, api.AgentError) as error:
                raise ValueError(
                    "--receipt-log needs a real parent and a readable, unshared JSONL file"
                ) from error

        def put(self, receipt):
            if self.get(receipt.tool_call_id) is not None:
                return super().put(receipt)
            try:
                with _open_log(policy.root, raw, append=True) as output:
                    output.write(json.dumps(receipt.to_dict(), sort_keys=True) + "\n")
            except (OSError, ValueError) as error:
                raise api.AgentError("could not append the CLI receipt log") from error
            return super().put(receipt)

    def normalized_path(path: Path) -> str:
        return unicodedata.normalize("NFD", str(path)).casefold()

    log_key = normalized_path(log_path)

    def is_receipt(path: Path) -> bool:
        if normalized_path(path) == log_key:
            return True
        try:
            target, log = path.lstat(), log_path.lstat()
        except FileNotFoundError:
            return False
        except OSError as error:
            raise api.AgentError(
                "could not check the reserved CLI receipt path"
            ) from error
        return (target.st_dev, target.st_ino) == (log.st_dev, log.st_ino)

    class ReceiptWorkspace(api.Workspace):
        def resolve_path(self, raw_path, *, must_exist=True):
            path = super().resolve_path(raw_path, must_exist=must_exist)
            if is_receipt(path):
                raise api.AgentError("the CLI receipt log is reserved")
            return path

        def list_files(self, raw_path="."):
            listing = super().list_files(raw_path)
            visible = [
                line
                for line in listing.splitlines()
                if not line.startswith("file ")
                or not is_receipt(policy.root / line.removeprefix("file "))
            ]
            return "\n".join(visible) or "(empty directory)"

    return ReceiptWorkspace(policy, confirm, CliReceiptStore()), log_path
