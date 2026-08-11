# Day 1: A Validated Agent Loop

> **Day 1 scope:** This chapter teaches a bounded loop and a JSON action
> protocol. The supplied test uses a fake read-only workspace. File mutation
> and command execution are not Day 1 capabilities.

A text generator returns one response and stops. A coding agent needs a small
control loop: it asks for one response, decides whether that response is a
final answer or an action, records what happened, and repeats when an action
produces an observation.

The model never edits a file directly. It emits text. Ordinary Python code
validates that text before handing a parsed action to the workspace object.
That separation is what makes the loop testable without a model.

## Files and Commands

Implement these Day 1 starter functions:

| File | Function or type | Your responsibility |
| --- | --- | --- |
| `src/tiny_llm/agent/generation.py` | `initial_messages()`, `generate_response()` | Reject a blank task, create the first messages, and decode one response with a fresh cache. |
| `src/tiny_llm/agent/protocol.py` | `AgentError`, `FinalAction`, `ToolAction`, `TOOL_CATALOG_HASH`, `tool_catalog_hash()`, `parse_action()`, `build_system_prompt()` | Define the exact action vocabulary, validate one JSON object, and describe only enabled actions. |
| `src/tiny_llm/agent/loop.py` | `AgentLimits`, `AgentEvent`, `AgentRun`, `run_agent()` | Bound the loop, append observations, and return an auditable result. |

Run the focused learner check:

```bash
pdm run test --week 4 --day 1
```

It should pass without loading a model. For the supplied reference check:

```bash
pdm run test-refsol --week 4 --day 1
```

`generate_response()` is still a Day 1 public boundary even though the focused
test deliberately avoids model weights. Render the messages with the tokenizer,
decode at most the requested token count using a fresh cache, stop at EOS, and
release every cache in a finally block. The scripted loop tests are the fast
way to verify the control flow; a real model is not required for this checkpoint.

## One Response, One Structured Decision

Use JSON because it makes the protocol visible in a trace. A response is
either a final answer:

```json
{"final":"I inspected README.md."}
```

or a tool request:

```json
{"tool":"read_file","path":"README.md"}
```

`parse_action()` must accept exactly one JSON object. It rejects malformed
JSON, non-object values, an empty final answer, an unknown tool, disabled
tools, missing required fields, unexpected fields, and fields with the wrong
shape. Do not quietly ignore trailing or extra data.

`TOOL_FIELDS` names the complete future vocabulary:
`list_files`, `read_file`, `write_file`, `edit_file`, and `run_command`.
Day 1 does not implement those effects. Its fake workspace enables only
`read_file`, which is enough to prove that the loop validates availability
before dispatching an action.

## Start a Conversation Deliberately

`initial_messages(task, system_prompt)` creates the first two messages:

```python
[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": task},
]
```

Reject an empty or whitespace-only task. A clear first message lets later turns
grow from a known history instead of assembling prompt fragments ad hoc.

`build_system_prompt(workspace)` describes the enabled action set for this
run. The prompt is guidance, not enforcement: `parse_action()` and the
workspace boundary must independently reject anything the policy does not
allow.

## The Bounded Loop

`run_agent()` receives a task, a `generate` callable, and a workspace. The
tests substitute a callable that returns predetermined strings, so the loop's
behavior stays deterministic.

```python
messages = initial_messages(task, build_system_prompt(workspace))
for step in range(1, limits.max_steps + 1):
    response = generate(messages)
    action = parse_action(response, workspace.available_tools)

    if action is a final answer:
        record it and stop

    result = workspace.execute(action)
    record the action and result
    messages = append the assistant response and tool observation
```

The real implementation also turns a parse failure into an observation such
as `error: response is not valid JSON: ...`, then lets the model try again.
This is more useful than crashing the agent for one malformed answer.

Every interaction becomes an `AgentEvent` containing the step number, raw
response, parsed action when one exists, and result or validation error. The
returned `AgentRun` records whether a valid final answer completed the
protocol. It does **not** prove that a task was solved; task grading is a later
course concern.

## Stop Conditions Are Part of Correctness

`AgentLimits` requires positive values. Implement all of these terminal cases:

- a valid final answer returns `completed`;
- reaching `max_steps` returns `step_limit`;
- too many invalid actions returns `invalid_action_limit`;
- an overlong conversation returns `context_limit`; and
- too many identical tool requests returns `repeated_action_limit`.

The repeated-action check matters even when a tool succeeds. Repeating the
same request can burn the whole budget while adding no new information.

## Exercise Checklist

Before considering Day 1 complete, make the focused test demonstrate all of
these behaviors:

1. A task starts with a system message and a user message.
2. A `read_file` request reaches the fake workspace, its result becomes an
   observation, and a later final answer stops the run.
3. Invalid JSON and an unavailable tool become recoverable error observations.
4. The loop stops at the step budget.
5. Repeated identical actions stop before the general step budget is spent.

Keep the solution inside the Day 1 starter files. Later days add session,
checkpoint, and rewind behavior.

{{#include copyright.md}}
