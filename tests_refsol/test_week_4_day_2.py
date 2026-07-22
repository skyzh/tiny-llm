import pytest

from .tiny_llm_base import (
    AgentError,
    FinalAction,
    ToolAction,
    ToolPolicy,
    Workspace,
    build_system_prompt,
    parse_action,
)


def test_task_1_parse_final_and_tool_actions():
    assert parse_action('{"final":"done"}') == FinalAction("done")
    assert parse_action('{"tool":"read_file","path":"README.md"}') == ToolAction(
        "read_file", {"path": "README.md"}
    )


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"final":"done"} trailing text',
        '{"tool":"unknown"}',
        '{"tool":"read_file"}',
        '{"tool":"read_file","path":7}',
        '{"final":"done","tool":"read_file"}',
    ],
)
def test_task_1_rejects_malformed_actions(response):
    with pytest.raises(AgentError):
        parse_action(response)


def test_task_2_prompt_contains_only_enabled_tools(tmp_path):
    read_only = Workspace(ToolPolicy(tmp_path))
    prompt = build_system_prompt(read_only)
    assert '"tool":"read_file"' in prompt
    assert '"tool":"write_file"' not in prompt
    assert '"tool":"run_command"' not in prompt

    writable = Workspace(ToolPolicy(tmp_path, allow_writes=True))
    prompt = build_system_prompt(writable)
    assert '"tool":"write_file"' in prompt
    assert '"tool":"edit_file"' in prompt


def test_task_2_rejects_a_known_but_disabled_tool(tmp_path):
    workspace = Workspace(ToolPolicy(tmp_path))
    with pytest.raises(AgentError, match="not enabled"):
        parse_action(
            '{"tool":"write_file","path":"x","content":"y"}',
            workspace.available_tools,
        )
