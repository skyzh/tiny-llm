import pytest

from .tiny_llm_base import AgentError, ToolAction, ToolPolicy, Workspace


def test_task_1_lists_and_reads_bounded_workspace_files(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")
    workspace = Workspace(ToolPolicy(tmp_path))

    listing = workspace.list_files()
    assert "file README.md" in listing
    assert ".env" not in listing
    assert workspace.read_file("README.md") == "hello"


@pytest.mark.parametrize("path", ["../outside", ".git/config", ".env", "key.pem"])
def test_task_2_rejects_unsafe_paths(tmp_path, path):
    workspace = Workspace(ToolPolicy(tmp_path))
    with pytest.raises(AgentError):
        workspace.resolve_path(path, must_exist=False)


def test_task_2_rejects_symlinks(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(target)
    workspace = Workspace(ToolPolicy(tmp_path))

    with pytest.raises(AgentError, match="symlinks"):
        workspace.read_file("link.txt")


def test_task_3_writes_are_disabled_by_default(tmp_path):
    workspace = Workspace(ToolPolicy(tmp_path))
    result = workspace.execute(
        ToolAction("write_file", {"path": "new.txt", "content": "data"})
    )

    assert result.startswith("error: writes are disabled")
    assert not (tmp_path / "new.txt").exists()
