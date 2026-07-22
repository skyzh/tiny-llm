# 🚧 Week 4: Build a Coding Agent

> 🚧 **Course status:** The daily chapters are drafts and are not included in
> the rendered book yet.

Weeks 1 through 3 turned tokens into text, made decoding efficient, and
introduced serving techniques. Week 4 adds the next layer: an agent loop that
lets the model observe a workspace, choose a tool, see the result, and continue
until a coding task is complete.

The goal is not to reproduce a production coding agent. It is to understand the
small mechanism underneath one and identify where reliability, efficiency, and
safety come from. By the end of the week, you will have a local CLI agent powered
by the inference stack you already built.

## What You Will Build

The finished agent can:

- inspect a repository without loading every file into the prompt;
- read files in bounded chunks and make exact, reviewable edits;
- run a narrowly scoped test command and use its output as feedback;
- save and resume a session;
- compact an overlong context while retaining task state;
- checkpoint and undo its own file mutations;
- accept steering messages and interrupt long-running work; and
- solve a small repository task graded by held-out tests.

This is a deliberately small target. Features such as multi-agent delegation,
remote execution, MCP integrations, and long-term user memory remain extensions
rather than prerequisites.

## The Core Loop

Every chapter builds on the same loop:

1. Render the task, project instructions, recent events, and tool descriptions.
2. Decode one structured action using the model and KV cache.
3. Parse and validate the action before it reaches the operating system.
4. Run one workspace tool and append its observation to the session.
5. Repeat until the model returns a final answer or reaches a budget.

The model does not edit files directly. It proposes an action; ordinary code
decides whether that action is valid and performs it. This boundary makes agent
behavior inspectable and testable.

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

## A Small Tool Surface

The target agent uses four tools inspired by small coding-agent harnesses:

```text
read(path, offset?, limit?)
edit(path, old_text, new_text)
write(path, content)
bash(command, timeout?)
```

`read` and `edit` are preferable to shell equivalents because they can enforce
consistent bounds and return structured errors. `bash` supplies repository
search, file discovery, and test execution without requiring a separate tool for
every command-line program.

A shell working directory is not a security sandbox. During this course, run the
agent only in a disposable exercise workspace. A production agent would need a
container, virtual machine, or similarly strong isolation boundary.

## Seven-Day Plan

| Day | Topic | Working milestone |
| --- | --- | --- |
| 1 | Agent loop | The model alternates between actions and observations. |
| 2 | Tools | The agent can read, edit, write, and run a bounded command. |
| 3 | Safety and validation | Mutations are confined, reviewable, and followed by validation. |
| 4 | Sessions | Work survives process exit and can be resumed. |
| 5 | Compaction | Long sessions retain a small, useful working context. |
| 6 | Control and recovery | The user can steer, interrupt, checkpoint, and undo. |
| 7 | Evaluation | The agent fixes a small bug and passes held-out tests. |

## Run the Starting Demo

The repository contains a minimal demonstration that uses the reference model
implementation:

```bash
pdm run agent "inspect this project and summarize its files"
```

Pass `--solution tiny_llm` to use your implementation, or `--solution mlx` to
use MLX-LM's optimized executor. The starting program is intentionally smaller
than the final agent: each day replaces one shortcut with an explicit component
that can be inspected and tested.

The default Qwen3 4B model follows the structured action protocol more reliably.
Use `--model qwen3-0.6b` on memory-constrained machines and expect to spend more
time on malformed-action recovery.

## Milestones

- **Minimal:** the model can inspect a workspace and produce one valid action.
- **Useful:** the agent can make a precise change and run its test.
- **Recoverable:** the session can resume, compact, and undo its own changes.
- **Controllable:** budgets bind and the user can steer or interrupt work.
- **Measurable:** a repeatable task suite distinguishes progress from anecdotes.

## Further Reading

- [Pi coding agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)
- [Benchmarking Coding Agents on Databricks' Multi-Million Line Codebase](https://www.databricks.com/blog/benchmarking-coding-agents-databricks-multi-million-line-codebase)

{{#include ../copyright.md}}
