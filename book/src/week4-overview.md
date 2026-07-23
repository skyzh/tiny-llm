# 🚧 Week 4: Build a Coding Agent

> 🚧 This overview and its daily chapters are under review and may change.

Weeks 1 through 3 built the inference stack that turns tokens into text and
serves multiple requests. Week 4 treats that stack as a generation backend and
adds an application layer: a coding agent that observes a workspace, proposes
one action, receives the result, and continues until it finishes or reaches a
budget.

The goal is a small system whose behavior you can inspect. It is not a production
agent or an operating-system sandbox.

## Learning Objectives

By the end of the week, you should be able to:

- separate model generation, action validation, tool execution, and session
  state;
- expose a small set of bounded repository tools;
- make file mutations explicit, bounded, and reviewable;
- preserve useful task state when a conversation grows too long;
- stop an agent with step, token, context, and tool limits; and
- evaluate a repository task by its result and side effects, not by how
  convincing the model's narration sounds.

Features such as multi-agent delegation, remote execution, MCP integrations, and
long-term user memory remain optional extensions.

## Prerequisites

Before starting Week 4:

1. Complete the Week 2 model, including its request-scoped KV cache, or use the
   reference solution.
2. Be able to run one generation through `pdm run main --loader week2`.
3. If you want to serve concurrent agent sessions, complete the Week 3 scheduler
   and paged-cache chapters first.
4. Create a disposable exercise workspace. A working-directory check is not a
   security boundary, so do not point the agent at sensitive files or a checkout
   that you cannot replace.

Start by confirming that the reference agent can inspect a workspace in read-only
mode:

```bash
pdm run agent --root . "inspect this project and summarize its files"
```

The CLI defaults to the reference Week 2 backend. Use `--solution tiny_llm` to
exercise your implementation, `--loader week3` to use the Week 3 model, or
`--solution mlx` to use MLX-LM as the executor. Qwen3-4B follows the structured
action protocol more reliably; `--model qwen3-0.6b` uses less memory but produces
more malformed actions.

## The Agent Boundary

Keep the model below a narrow application interface. It receives messages and
returns text. Ordinary code parses that text, decides whether an action is
allowed, executes it, and records the observation:

```text
task + session events
        |
        v
  context builder ---> model ---> action validator
        ^                            |
        |                            v
   tool result <--- tool runner <--- validated action
                         |
                         v
                      workspace
```

The target JSON tool surface is intentionally small:

```text
list_files(path?)
read_file(path)
write_file(path, content)
edit_file(path, old, new)
run_command(argv)
```

Prefer `list_files`, `read_file`, and `edit_file` to shell equivalents because
they enforce consistent bounds and return structured errors. Treat
`run_command` as an explicitly authorized escape hatch for repository search
and validation; accept a non-empty argument array rather than an interpolated
shell string.

## Seven-Day Plan

Each day should end with one observable checkpoint. Do not add the next
capability until the current one has a focused test.

| Day | Task | Checkpoint |
| --- | --- | --- |
| 1 | Build initial messages and connect generation. | The task and system instructions reach a fresh model conversation. |
| 2 | Parse structured actions and expose only enabled tools. | Malformed or disabled actions fail before execution. |
| 3 | Add bounded listing and reading. | Paths, secrets, symlinks, and default read-only policy are enforced. |
| 4 | Append observations and compact context. | The task anchors and newest complete tool turn survive compaction. |
| 5 | Add writes, exact edits, and authorized commands. | Existing files must be inspected, edits match once, and commands match an exact allowlist. |
| 6 | Run the bounded recovery loop. | The agent recovers from invalid output and stops repeated actions. |
| 7 | Evaluate a held-out repository task. | The expected tests pass and forbidden side effects are absent. |

The coding-agent chapter develops the loop and tool boundary. The RAG chapter
adds retrieved evidence before generation. The tool-calling and serving chapter
moves durable agent state onto the Week 3 serving interfaces without changing
the model kernels.

## Validation Strategy

Validate the layer you just added before running an end-to-end demo:

- parser tests cover malformed JSON, unknown tools, missing arguments, and extra
  arguments;
- workspace tests cover path traversal, symlinks, protected files, size limits,
  and uninspected overwrites;
- loop tests cover step limits, repeated invalid actions, repeated identical
  actions, and model/tool errors;
- context tests ensure compaction keeps the original task and complete recent
  action/observation turns; and
- capstone tests check the requested result, modified paths, tool trace, and
  forbidden side effects.

Run the focused test for the day you are implementing:

```bash
pdm run test --week 4 --day <N>
```

Then run the CLI in read-only mode. Enable mutations only in a disposable
workspace, and authorize validation commands one exact argument vector at a
time.

## Further Reading

- [Pi coding agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)
- [Benchmarking Coding Agents on Databricks' Multi-Million Line Codebase](https://www.databricks.com/blog/benchmarking-coding-agents-databricks-multi-million-line-codebase)

{{#include copyright.md}}
