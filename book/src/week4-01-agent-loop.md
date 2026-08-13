# Day 1: A Validated Agent Loop

Weeks 1 through 3 built a function that turns a conversation into model text.
A coding agent needs a small control loop around that function: ask for one
response, decide whether it is a final answer or an action, record what
happened, and continue when an action produces an observation.

The model never edits a file directly. It emits text. Ordinary Python validates
that text before handing a parsed action to a workspace object. This separation
makes the loop deterministic to test even when no model weights are loaded.

## The Teaching Boundary

Day 1 teaches only a bounded loop and one JSON action protocol. The supplied
test uses a fake workspace with one enabled read-only action. Real project
inspection arrives on Day 2; file mutation, command execution, approval, and
durable receipts arrive later.

The loop validates and records a model response, but it does not prove that the
model solved the task. It is also not a sandbox, background worker, persistent
session, or production scheduler.

## Files and Public Surface

Implement the TODO bodies in these Day 1 starter files:

| File | Public names | Responsibility |
| --- | --- | --- |
| `src/tiny_llm/agent/generation.py` | `initial_messages`, `generate_response` | Begin a conversation and keep one model-response boundary explicit. |
| `src/tiny_llm/agent/protocol.py` | `AgentError`, `FinalAction`, `ToolAction`, `parse_action`, `build_system_prompt` | Represent and validate one final answer or one enabled tool request. |
| `src/tiny_llm/agent/loop.py` | `AgentLimits`, `AgentEvent`, `AgentRun`, `run_agent` | Bound a run, propagate observations, and retain an inspectable trace. |

`generate_response()` remains part of the public Day 1 surface even though the
focused test uses scripted strings. It renders the messages with the course
tokenizer, decodes at most `max_tokens` with a fresh cache, stops at EOS, and
releases every cache in a `finally` block.

Run the cumulative learner checkpoint from the repository root:

```bash
pdm run test --week 4 --day 1
```

This command copies the supplied Day 1 test into `tests/` before running it.
Before you implement the TODOs, the implementation-dependent cases across nine
task groups are expected to fail. No model download is required.

Course maintainers can check the supplied implementation without copying the
learner test:

```bash
pdm run test-refsol --week 4 --day 1
```

## Task 1: Start the Conversation Deliberately

`initial_messages(task, system_prompt)` creates the first two messages:

```python
[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": task},
]
```

Reject an empty or whitespace-only task. `build_system_prompt(workspace)`
describes only the actions enabled for this run. The prompt is guidance, not
enforcement: the protocol and workspace boundary must still reject anything
the policy does not allow.

## Task 2: Execute One Tool and Finish

`run_agent(task, generate, workspace, limits=None)` starts from those messages.
The test injects a `generate` callable that returns predetermined strings, so
the control flow stays deterministic.

For a valid tool action, call `workspace.execute(action)`, record the action and
result, and append both the assistant response and a user observation. When a
later response is a valid `FinalAction`, return a completed `AgentRun` with the
final text.

## Task 3: Validate One Structured Decision

A model response is exactly one JSON object. It is either a final answer:

```json
{"final":"I inspected README.md."}
```

or one tool request:

```json
{"tool":"read_file","path":"README.md"}
```

`parse_action()` rejects malformed JSON, non-object values, blank final text,
unknown or disabled tools, missing fields, unexpected fields, and fields with
the wrong shape. Do not ignore trailing or extra data.

`TOOL_FIELDS` names the cumulative vocabulary: `list_files`, `read_file`,
`write_file`, `edit_file`, and `run_command`. Day 1 implements none of those
effects. Its fake workspace enables only `read_file`, which is enough to prove
that availability is checked before dispatch.

Malformed or unavailable actions become ordinary `error:` observations. The
model can see the failure and choose another response instead of crashing the
Python loop.

## Task 4: Stop at the Step Budget

`AgentLimits.max_steps` bounds how many model decisions one run may attempt.
When the loop consumes that budget without a valid final answer, return an
incomplete run whose reason is `step_limit`. The events show exactly how the
budget was spent.

## Task 5: Return an Inspectable Run

Every interaction becomes an `AgentEvent` with the step number, raw response,
parsed action when one exists, and result or validation error. `AgentRun`
records the completion flag, stop reason, optional final answer, and immutable
event tuple.

A run marked `completed` means only that the model returned a valid final
action. Later days add receipts and outcome evaluation; Day 1 keeps the trace
small and in memory.

## Task 6: Recover from Invalid JSON

After an invalid response, append the raw assistant response and its exact
validation error as the next user observation. Reset the identical-action
counter, then let the model try again while the invalid-action budget remains.

The focused case sends invalid JSON followed by a valid final response. The
first event must retain the recoverable error, and the second must complete the
run.

## Task 7: Stop Repeated Actions

Serialize each parsed tool name and normalized argument object into a stable
signature. Count consecutive identical requests and stop with
`repeated_action_limit` when the count exceeds the configured budget.

This guard matters even when a tool succeeds: repeating the same request can
consume the whole run without adding new information.

## Task 8: Preserve the Exact Observation

The next model call must receive the complete tool result, not only a marker:

```python
{
    "role": "user",
    "content": "Tool result:\nREADME contents",
}
```

The normal guard fails if that payload is changed or dropped. It also proves
that a known-but-disabled tool such as `write_file` becomes an error
observation and never reaches the fake workspace.

## Task 9: Make Every Limit Fail Closed

Require positive values for `max_steps`, `max_context_chars`,
`max_invalid_actions`, and `max_identical_actions`. Zero or negative budgets
would disable the intended stopping guarantee and must be rejected.

Before each model call, bound the total message-content characters and stop
with `context_limit` when it is too large. Count invalid actions and stop with
`invalid_action_limit` when that budget is exhausted. Together with the step
and repeated-action limits, every run has an explicit terminal reason.

## Checkpoint

When Day 1 is green, inspect the focused test rather than only its final pass:
confirm the initial system/user pair, one dispatched `read_file`, the exact
observation in the next model input, the completed final event, and each
budgeted stop reason.

You now have a validated, bounded model → action → observation loop. Continue
with [Day 2: Inspect a Workspace](week4-02-tools.md) to replace the fake tool
boundary with real contained directory listing and UTF-8 file reads.

{{#include copyright.md}}
