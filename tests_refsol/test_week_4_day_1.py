from .tiny_llm_base import initial_messages


def test_task_1_initial_messages_preserve_the_task():
    messages = initial_messages("inspect the project", "system instructions")

    assert messages == [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "inspect the project"},
    ]


def test_task_1_rejects_an_empty_task():
    try:
        initial_messages("   ", "system instructions")
    except ValueError as error:
        assert "must not be empty" in str(error)
    else:
        assert False, "expected an empty task to be rejected"
