# 🚧 Week 4: Build the Coding Agent

> 🚧 This chapter is under review and may change.

This chapter builds the smallest complete coding-agent loop. Keep model
generation, action parsing, workspace access, and orchestration separate so each
boundary can be tested without running the whole agent.

## Objectives

By the end of this chapter, you should have an agent that can:

- turn a task into an initial conversation;
- generate and parse one structured action at a time;
- execute only tools allowed by an explicit workspace policy;
- return tool errors to the model as observations; and
- stop with a final answer or a bounded failure reason.

## Prerequisites

- Complete the Week 2 generation path or use `tiny_llm_ref`.
- Read the Week 4 overview and create a disposable workspace.
- Keep the agent above the model boundary: application code may request a fresh
  KV cache, but it must not reach into attention or kernel internals.

## Task 1: Connect Generation

Implement the message and generation helpers in the agent package. Start with a
system message and the user's task, render them with the tokenizer's chat
template, allocate a fresh request-scoped cache, and return generated text.

Checkpoint: two calls with separate tasks must not share conversation or KV-cache
state.

```bash
pdm run test --week 4 --day 1
```

## Task 2: Define Structured Actions

Define one schema for a final answer and one for a tool request. Parse exactly one
action from the model response and reject malformed JSON, unknown action shapes,
unknown tools, and arguments that do not match the selected tool.

Build the system prompt from the tools enabled for this run. Do not advertise a
write or command tool that the policy has disabled.

Checkpoint: valid actions round-trip through the parser, and every invalid case
returns a recoverable error rather than executing a tool.

```bash
pdm run test --week 4 --day 2
```

## Task 3: Add a Workspace Boundary

Implement bounded file listing and reading first. Resolve every path relative to
one explicit root and reject absolute paths, parent traversal, symlink escapes,
repository metadata, common secret files, and oversized reads.

Add writes only after the read-only path works. Require a prior read before
replacing an existing file, perform exact edits, and write atomically. Keep
command execution disabled unless the operator authorizes the exact argument
vector.

Checkpoints:

```bash
pdm run test --week 4 --day 3
pdm run test --week 4 --day 5
```

## Task 4: Run a Bounded Loop

On each step:

1. build the current model context;
2. generate and parse one action;
3. stop if it is a final answer;
4. otherwise execute the validated tool;
5. append the complete action/observation turn; and
6. stop when a configured budget is exhausted.

Return parse and tool failures to the model so it can recover. Bound total steps,
invalid responses, repeated actions, generated tokens, and context size.

```bash
pdm run test --week 4 --day 4
pdm run test --week 4 --day 6
pdm run test --week 4 --day 7
```

## Validate the Chapter Checkpoint

Run the reference agent in read-only mode first:

```bash
pdm run agent --root . "inspect this project and summarize its files"
```

Next, create a disposable exercise directory and enable only the capability under
test. For example, permit writes without enabling commands:

```bash
pdm run agent --root ./exercise --allow-writes \
  "read README.md and make one requested exact edit"
```

Inspect the event trace and modified paths. A successful run should have a clear
stop reason, no access outside the root, and no side effect that was not required
by the task.

{{#include copyright.md}}
