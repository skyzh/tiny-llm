from .tiny_llm_base import ToolPolicy, Workspace, run_agent


def responses(*items):
    """Test helper for Week 4, Day 7: return scripted model responses."""

    queue = iter(items)
    return lambda messages: next(queue)


def test_capstone_records_a_safe_reviewable_change(tmp_path):
    source = tmp_path / "answer.py"
    source.write_text("ANSWER = 41\n", encoding="utf-8")
    workspace = Workspace(ToolPolicy(tmp_path, allow_writes=True))
    generate = responses(
        '{"tool":"read_file","path":"answer.py"}',
        '{"tool":"edit_file","path":"answer.py","old":"41","new":"42"}',
        '{"final":"updated the answer"}',
    )

    result = run_agent("fix the answer", generate, workspace)

    assert result.completed
    assert result.modified_files == ("answer.py",)
    assert source.read_text(encoding="utf-8") == "ANSWER = 42\n"
    assert [event.step for event in result.events] == [1, 2, 3]


def test_capstone_cannot_smuggle_in_a_command(tmp_path):
    workspace = Workspace(ToolPolicy(tmp_path))
    generate = responses(
        '{"tool":"run_command","argv":["rm","-rf","."]}',
        '{"final":"command was refused"}',
    )

    result = run_agent("inspect the project", generate, workspace)

    assert result.completed
    assert result.events[0].action is None
    assert "not enabled" in result.events[0].result
