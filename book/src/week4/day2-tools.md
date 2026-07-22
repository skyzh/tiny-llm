# Day 2: Tools and Structured Actions

The agent loop becomes useful when actions can inspect and change a repository.
Today you will build a small tool surface modeled after practical coding agents:
`read`, `edit`, `write`, and `bash`.

## Why Four Tools Are Enough

A larger tool catalog is not automatically more capable. Every schema consumes
context and gives the model another choice to make. Four tools cover the basic
software-development cycle:

- `read` gathers exact source context;
- `edit` makes a targeted, reviewable replacement;
- `write` creates a file or deliberately replaces its contents; and
- `bash` handles discovery, search, formatting, compilation, and tests.

The model can use `rg`, `find`, and `ls` through `bash`. Later you can add
specialized tools when evaluation shows that a repeated workflow is unreliable,
not merely because another command exists.

## Read

Use a line-oriented interface:

```json
{"tool":"read","path":"src/parser.py","offset":40,"limit":80}
```

The result should state which lines were returned and how to continue. Apply both
a line limit and a byte limit so a minified or binary-looking file cannot flood
the context. Reject directories and unreadable or non-text files with explicit
errors.

## Edit

Use exact text replacement rather than asking the model to rewrite an entire
file:

```json
{
  "tool":"edit",
  "path":"src/parser.py",
  "old_text":"if not value:\n    return None",
  "new_text":"if value is None:\n    return None"
}
```

The old text must match exactly once. Zero matches usually mean the model needs
to reread the file. Multiple matches mean it must select a more specific region.
On success, return a unified diff rather than the complete new file.

An exact edit has useful failure semantics: it refuses to apply when the file no
longer matches what the model observed.

## Write

`write` is useful for creating a new file, but overwriting an existing file is a
larger mutation than an exact edit. Make that distinction visible in policy and
in the trace. Enforce a content limit and use an atomic replace so interruption
does not leave a partially written file.

## Bash

The input contains a command and an optional timeout:

```json
{"tool":"bash","command":"pdm run test --week 1 --day 3","timeout":120}
```

Capture stdout and stderr together in execution order, include the exit code,
and cap what is returned to the model. Preserve the complete output in a
temporary log when truncation occurs and tell the model where the retained tail
begins.

The timeout must terminate the process tree, not just the shell parent. Day 6
will use the same cancellation path for interactive interrupts.

> A command running with `cwd` set to the repository can still read files outside
> that directory. Run this course agent only in a disposable workspace. Path
> validation is useful, but it is not a replacement for process isolation.

## Tool Registry

Avoid a long `if`/`elif` chain by representing tools as data:

```python
TOOLS = {
    "read": Tool(schema=ReadArgs, execute=read_file),
    "edit": Tool(schema=EditArgs, execute=edit_file),
    "write": Tool(schema=WriteArgs, execute=write_file),
    "bash": Tool(schema=BashArgs, execute=run_command),
}
```

The same registry can render tool descriptions for the prompt, validate model
arguments, dispatch execution, and expose tool-specific risk metadata.

## Exercise

Implement the four tools and test them without a model:

1. Read a large file in two windows.
2. Reject an edit with zero or multiple matches.
3. Apply a unique edit while preserving the file's line endings.
4. Create a new file and reject content over the configured limit.
5. Run a command that succeeds, one that fails, and one that times out.
6. Truncate a large command result while preserving its exit status.

Then give the agent a fixture repository containing a one-line bug. A successful
trajectory should inspect the implementation and test, make one exact edit, run
the focused test, and return a final answer.

{{#include ../copyright.md}}
