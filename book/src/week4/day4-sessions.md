# Day 4: Sessions and Resume

So far, exiting the process loses the agent's conversation and progress. Today
you will store an append-only session that can be inspected, resumed, and later
compacted without treating a rendered prompt as the source of truth.

## Three Kinds of State

Keep these states separate:

1. **Durable session state** records user messages, model actions, tool results,
   checkpoints, and control events.
2. **Model-visible context** is the selected and rendered subset sent on the next
   model turn.
3. **KV-cache state** is an optimization tied to one token sequence and model
   instance.

A session remains valid after its KV cache is evicted. A context can be compacted
without deleting the durable trace. This separation also lets the agent resume
after a process restart by rebuilding model context from stored events.

## Append-Only JSONL

Use one JSON object per line. Every event has an ID, timestamp, type, and
type-specific payload:

```json
{"id":"01...","type":"user_message","content":"fix empty values"}
{"id":"02...","type":"assistant_message","content":"{...}"}
{"id":"03...","type":"tool_call","tool":"read","arguments":{"path":"src/config.py"}}
{"id":"04...","type":"tool_result","tool_call_id":"03...","is_error":false,"content":"..."}
```

Append and flush an assistant response before executing its action. If the
process exits during a tool call, resume logic can detect the unmatched call and
record an interruption rather than pretending it completed.

Do not store secrets copied from the process environment. Treat session files as
sensitive because source excerpts and command output may themselves contain
private information.

## Session Lifecycle

Support a minimal command set:

```text
agent TASK                 start a new session
agent --continue           continue the newest session for this workspace
agent --session SESSION_ID resume a selected session
agent --no-session TASK    run without persistence
```

Store the resolved workspace path and model identifier in session metadata.
Refuse to resume in a different workspace unless the user explicitly selects a
new root; silently replaying file actions against another repository is unsafe.

## Rebuilding a Conversation

The context builder maps events back to chat messages:

- user events become user messages;
- assistant responses remain assistant messages;
- tool results use the role or wrapper expected by the model's chat template;
- checkpoints and timing metadata remain in the durable log but need not be
  shown to the model; and
- interrupted tool calls become concise error observations.

Rendering happens only when a model turn starts. This prevents presentation
details from becoming the persistent data model.

## Project Instructions

At session start, discover an `AGENTS.md` or equivalent instruction file in the
workspace and add its contents to the initial context. Record which instruction
files were loaded so resume behavior is explainable.

Choose a policy for changed instructions. A simple implementation rereads them
when resuming and records an event if their content hash differs.

## Serving Multiple Agent Sessions

The Week 3 scheduler may run model requests for several agents, but each agent
still owns its own event stream, context budget, and tool execution state. Keep
the scheduler interface narrow:

```text
session context -> generation request -> assistant action
```

The serving layer should not write session files or execute tools. It may evict a
session's KV cache while the durable event log remains resumable.

## Exercise

1. Serialize and reload every event type without losing fields.
2. Stop an agent after two tool calls, restart the program, and continue it.
3. Simulate a crash between a tool call and its result.
4. Verify that `--no-session` creates no durable file.
5. Attempt to resume a session from a different workspace.
6. Change `AGENTS.md` between runs and make the change visible in the trace.

At the end of the day, a model should be able to resume the bug-fixing task from
Day 3 without restarting the reasoning process from the original prompt.

{{#include ../copyright.md}}
