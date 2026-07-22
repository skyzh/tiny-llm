from .tiny_llm_base import AgentLimits, ToolPolicy, Workspace, run_agent


def responses(*items):
    """Test helper for Week 4, Day 6: return scripted model responses."""

    queue = iter(items)
    return lambda messages: next(queue)


def test_task_1_agent_observes_then_finishes(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    workspace = Workspace(ToolPolicy(tmp_path))
    generate = responses(
        '{"tool":"read_file","path":"README.md"}',
        '{"final":"inspected README"}',
    )

    result = run_agent("inspect the project", generate, workspace)

    assert result.completed
    assert result.final == "inspected README"
    assert result.reason == "completed"
    assert len(result.events) == 2


def test_task_2_agent_recovers_from_invalid_json(tmp_path):
    workspace = Workspace(ToolPolicy(tmp_path))
    generate = responses("not json", '{"final":"recovered"}')

    result = run_agent("inspect the project", generate, workspace)

    assert result.completed
    assert result.events[0].result.startswith("error:")


def test_task_3_agent_stops_repeated_actions(tmp_path):
    workspace = Workspace(ToolPolicy(tmp_path))
    generate = responses(*['{"tool":"list_files"}'] * 3)

    result = run_agent(
        "inspect the project",
        generate,
        workspace,
        AgentLimits(max_steps=5, max_identical_actions=2),
    )

    assert not result.completed
    assert result.reason == "repeated_action_limit"
