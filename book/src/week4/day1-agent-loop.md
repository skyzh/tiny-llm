# Day 1: From Generation to an Agent Loop

The decoder from Week 1 produces text once and exits. An agent repeatedly turns
text into an action, executes that action, and gives the result back to the
model. Today you will make that control flow explicit.

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
{"tool":"read","path":"README.md"}
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
error: read requires a non-empty string field named path
```

## The Loop

Keep orchestration separate from inference and tool execution:

```python
def run_agent(task, model, tools, max_steps):
    events = [user_message(task)]
    for _ in range(max_steps):
        response = model.generate(events, tools)
        events.append(assistant_message(response))

        action = parse_and_validate(response)
        if action.is_final:
            return action.final

        result = tools.execute(action)
        events.append(tool_result(action, result))

    raise StepLimitExceeded(max_steps)
```

The loop owns policy such as budgets and stop conditions. The model adapter owns
tokenization and decoding. The tool registry owns schemas and execution. These
boundaries will matter when sessions and cancellation arrive later in the week.

## Preserve the Trace

For now, an in-memory list is sufficient. Record at least:

- the user's task;
- the assistant's raw response;
- the parsed action, when valid;
- the tool result or validation error; and
- token counts and elapsed time if they are available.

Do not store only the latest prompt string. Named events are easier to inspect
and can later be serialized without reverse-engineering the chat template.

## Exercise

Refactor the starting demo so `generate()` returns model text and a new
`run_agent()` function owns the loop.

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

At the end of Day 1, the following trace should be possible:

```text
user      inspect the repository
assistant {"tool":"read","path":"README.md"}
tool      # Tiny LLM ...
assistant {"final":"This repository teaches LLM inference and serving."}
```

The tool can still be a stub. Tomorrow it will become a real workspace API.

{{#include ../copyright.md}}
