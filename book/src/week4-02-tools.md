# Day 2: Tools and Structured Actions

> 🚧 **Early-review WIP:** This chapter is public for early review and may
> change. Use a disposable workspace when running the agent or enabling writes
> or commands.

The agent loop becomes useful when actions can inspect and change a repository.
The course protocol uses five actions: `list_files`, `read_file`, `edit_file`,
`write_file`, and `run_command`.

> **Implementation status:** The current Day 2 checkpoint covers strict action
> parsing and a system prompt that lists only enabled tools. Read-side workspace
> behavior is checked on Day 3; mutation and command behavior is checked on Day
> 5. The sections below describe that cumulative baseline using its exact JSON
> names and fields.

## Why Five Tools Are Enough

A larger tool catalog is not automatically more capable. Every schema consumes
context and gives the model another choice to make. Five tools cover the basic
software-development cycle:

- `list_files` performs bounded directory discovery;
- `read_file` gathers one bounded UTF-8 file;
- `edit_file` makes a targeted, reviewable replacement;
- `write_file` creates a file or deliberately replaces its contents; and
- `run_command` runs one exact operator-allowed argument vector.

The model can use `rg`, `find`, or another program only when the operator has
allowed that exact argument vector for `run_command`. Later you can add
specialized tools when evaluation shows that a repeated workflow is unreliable,
not merely because another command exists.

## List Files

The path is optional and defaults to the workspace root:

```json
{"tool":"list_files","path":"src"}
```

The result identifies entries as files or directories, omits protected paths and
symlinks, and stops after the policy's entry limit.

## Read File

The current interface reads one complete bounded file:

```json
{"tool":"read_file","path":"src/parser.py"}
```

Apply a byte limit so a minified or binary-looking file cannot flood the context.
Reject directories, oversized files, and non-UTF-8 files with explicit errors.
Line windows are a possible extension, but `offset` and `limit` are not fields in
the current action schema.

## Edit File

Use exact text replacement rather than asking the model to rewrite an entire
file:

```json
{
  "tool":"edit_file",
  "path":"src/parser.py",
  "old":"if not value:\n    return None",
  "new":"if value is None:\n    return None"
}
```

The old text must match exactly once. Zero matches usually mean the model needs
to reread the file. Multiple matches mean it must select a more specific region.
The current baseline returns the changed path; adding a bounded unified diff to
the trace is a planned reviewability improvement.

An exact edit has useful failure semantics: it refuses to apply when the file no
longer matches what the model observed.

`edit_file` is available only when writes were enabled by the operator. Even
then, each model-dispatched action through `Workspace.execute()` must pass its
preflight before it pauses and asks `y/N`. Only an explicit yes advances to the
post-approval recheck and atomic edit. Direct method calls are the trusted,
model-free layer used by focused tool tests and do not prompt.

## Write File

`write_file` is useful for creating a new file, but overwriting an existing file
is a larger mutation than an exact edit. An existing file must first be read.
Enforce a content limit and use an atomic replace so interruption does not leave
a partially written file. Like `edit_file`, every eligible model-dispatched
action that passes preflight requires an explicit `y`/`yes`; the displayed
default is No.

## Run Command

The input is a non-empty JSON array. It must exactly equal an argument vector the
operator supplied with `--allow-command`:

```json
{"tool":"run_command","argv":["pdm","run","test","--week","1","--day","3"]}
```

Capture stdout and stderr together in execution order, include the exit code,
and cap what is returned to the model. The current baseline retains the bounded
prefix and appends a truncation marker; it does not preserve a complete temporary
log.

The timeout belongs to `ToolPolicy`, not the model action. It attempts to
terminate the command's same-process-group descendants and reaps the foreground
child. A descendant that creates another session can escape this cleanup, so a
disposable workspace and stronger OS process isolation are still required for
hostile commands. At the Day 2 checkpoint, the runner handles
`KeyboardInterrupt` while a command is active. The cumulative reference
implementation adds one cooperative cancellation signal across the loop,
course-model decoder, command polling, and steering in Day 6; that behavior is
not part of the Day 2 focused check.

Exact allowlisting is necessary but not sufficient. Every eligible
model-dispatched `run_command` action that passes preflight also asks `y/N` and
defaults to No. The runner does not invoke a shell, but an allowed executable can
itself delete files, read outside the workspace, spawn children, or use the
network.

> A command running with `cwd` set to the repository can still read files outside
> that directory. Run this course agent only in a disposable workspace. Path
> validation is useful, but it is not a replacement for process isolation.

## Executable Schema

The current parser represents required and optional fields as data:

```python
TOOL_FIELDS = {
    "list_files": (frozenset(), frozenset({"path"})),
    "read_file": (frozenset({"path"}), frozenset()),
    "write_file": (frozenset({"path", "content"}), frozenset()),
    "edit_file": (frozenset({"path", "old", "new"}), frozenset()),
    "run_command": (frozenset({"argv"}), frozenset()),
}
```

Keep the prompt, parser, availability policy, approval policy, and dispatch names
in agreement. Unknown, disabled, malformed, or human-denied actions become
recoverable observations rather than operating-system calls.

## Current Checkpoint

Implement `parse_action()` and `build_system_prompt()` in
`src/tiny_llm/agent/protocol.py`, then run:

```bash
pdm run test --week 4 --day 2
```

The checks cover valid final and tool actions, malformed JSON and fields,
enabled-tool prompt rendering, and rejection of a known but disabled tool. Use
`pdm run test-refsol --week 4 --day 2` for the supplied implementation.

## Cumulative Tool Exercise

Implement the five tools and test them without a model across the later
checkpoints:

1. List a bounded directory and read a bounded UTF-8 file.
2. Reject an edit with zero or multiple matches.
3. Apply a unique approved edit while preserving the file's line endings.
4. Create an approved file and reject content over the configured limit.
5. Deny mutations and commands when approval is empty, unavailable, or No.
6. Run an exact approved command that succeeds, one that fails, and one that
   times out.
7. Truncate a large command result while preserving its exit status.

Then give the agent a fixture repository containing a one-line bug. A successful
trajectory should inspect the implementation and test, make one exact edit, run
the focused test, and return a final answer. This model-backed trajectory is a
course target; the current Day 2 tests are deterministic and do not load a model.

{{#include copyright.md}}
