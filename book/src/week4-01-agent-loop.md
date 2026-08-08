# Day 1: From Generation to an Agent Loop

> 🚧 **Early-review WIP:** This chapter is public for early review and may
> change. Use a disposable workspace when running the agent or enabling writes
> or commands.

The decoder from Week 1 produces text once and exits. An agent repeatedly turns
text into an action, executes that action, and gives the result back to the
model. Today you will make that control flow explicit.

> **Implementation status:** The current Day 1 learner check covers
> `initial_messages()`: it preserves the system instructions and task and rejects
> an empty task. The complete `run_agent()` loop described later in this chapter
> exists in the reference baseline, but its focused checks currently live under
> Day 6. Treat the rest of this chapter as the intended Day 1 expansion, not as
> behavior proved by the Day 1 test.

## Current Repository Checkpoint

Implement `initial_messages(task, system_prompt)` in
`src/tiny_llm/agent/generation.py`, then run:

```bash
pdm run test --week 4 --day 1
```

The expected result is that both Day 1 tests pass without loading a model. To
check the supplied implementation instead, run
`pdm run test-refsol --week 4 --day 1`.

## Learning Goals

By the end of the day, you will be able to:

- explain the difference between a model, an agent loop, and a tool;
- represent tool calls and final answers as structured actions;
- preserve assistant actions and tool observations in the conversation; and
- stop reliably on completion, malformed output, or a step budget.

## Actions, Not Free-Form Commands

Begin with a JSON protocol because it is visible in every trace and works with a
model that does not expose native tool calls. An assistant turn produces exactly
one of two shapes:

```json
{"tool":"read_file","path":"README.md"}
```

```json
{"final":"The project implements a small Qwen3 inference stack."}
```

Parsing JSON is only the first check. The decoded value must be an object, must
contain exactly one of `tool` or `final`, and must contain arguments allowed by
that action's schema. Reject trailing text rather than silently ignoring it.

Return validation failures to the model as observations. This lets the model
repair a malformed action without hiding the failure:

```text
error: missing fields for read_file: path
```

## The Loop

Keep orchestration separate from inference and tool execution:

```python
def run_agent(task, generate, workspace, limits):
    messages = initial_messages(task, build_system_prompt(workspace))
    events = []
    for step in range(1, limits.max_steps + 1):
        response = generate(messages)

        try:
            action = parse_action(response, workspace.available_tools)
        except AgentError as error:
            result = f"error: {error}"
            events.append(AgentEvent(step, response, None, result))
            messages = append_tool_result(messages, response, result)
            continue
        if isinstance(action, FinalAction):
            events.append(AgentEvent(step, response, action, None))
            break

        result = workspace.execute(action)
        events.append(AgentEvent(step, response, action, result))
        messages = append_tool_result(messages, response, result)

    # Construct AgentRun with either "completed" or "step_limit".
```

The loop owns policy such as budgets and stop conditions. The model adapter owns
tokenization and decoding. The tool registry owns schemas and execution. These
boundaries will matter when sessions and cancellation arrive later in the week.

## Preserve the Trace

For now, an in-memory list is sufficient. The current `AgentEvent` records the
step, raw response, parsed action when valid, and result. The target trace should
eventually also make these run-level facts discoverable:

- the user's task;
- the assistant's raw response;
- the parsed action, when valid;
- the tool result or validation error; and
- token counts and elapsed time if they are available.

Do not store only the latest prompt string. Named events are easier to inspect
and can later be serialized without reverse-engineering the chat template.

## Planned Loop Exercise

Keep `generate_response()` responsible for one model response and make
`run_agent()` own the loop.

Implement and test these cases without loading a model:

1. A valid tool action is executed and its result is appended.
2. A final action stops the loop without executing a tool.
3. Invalid JSON becomes a useful observation and the loop continues.
4. An unknown tool is rejected before execution.
5. The loop stops after `max_steps` even if the model never finishes.

Use a fake model that returns predetermined strings and a fake tool registry that
records calls. Most agent-loop behavior is ordinary deterministic code and does
not require expensive model tests.

## Checkpoint

After the planned loop slice is implemented, the following trace should be
possible:

```text
user      inspect the repository
assistant {"tool":"read_file","path":"README.md"}
tool      # Tiny LLM ...
assistant {"final":"This repository teaches LLM inference and serving."}
```

The tool can still be a stub in this planned slice. The current source tree's
bounded workspace API is introduced by the Day 2 and Day 3 checkpoints, and the
current loop behavior is verified by Day 6.

{{#include copyright.md}}
