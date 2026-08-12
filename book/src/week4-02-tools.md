# 🚧 Day 2: Authorize Effects and Keep Receipts

Day 1 stopped untrusted model text at a strict JSON boundary and bounded the
number of loop steps. Day 2 adds the next boundary: a model request is not
permission to change the machine.

You will build a workspace that confines file tools to one directory, asks the
operator before a write or command, and records each dispatched action as a
content-addressed receipt. The learner tests use temporary directories and
small local commands; they do not load a model.

> A path boundary is not a process sandbox. An allowed executable can read
> outside the workspace, use the network, or start other processes. Use a
> disposable repository and allow only exact commands you understand.

## The Day 1 to Day 2 Boundary

The Day 1 loop already calls `workspace.execute(action)`. Its fake workspace
proved the control flow without granting real effects. Day 2 replaces that fake
at runtime with two explicit layers:

```text
validated ToolAction
        |
        v
preflight against ToolPolicy
        |
        v
operator approval for writes or commands
        |
        v
revalidate, then perform the effect
        |
        v
append an EffectReceipt
```

Preflight must happen before approval. A malformed, disabled, stale, or
non-allowlisted action should never ask the operator a misleading question.
After approval, a write must recheck the file and parent directory before its
atomic replacement.

## Files and Public Surface

Implement the TODO bodies in these Day 2 starter files:

| File | Public names |
| --- | --- |
| `src/tiny_llm/agent/workspace.py` | `ToolPolicy`, `Workspace` |
| `src/tiny_llm/agent/receipts.py` | `EffectReceipt`, `ReceiptStore` |
| `src/tiny_llm/agent/__init__.py` | Re-exports the four Day 2 names alongside the Day 1 API. |

The starter declarations are the contract. Do not add extra public classes,
functions, constants, or methods.

`ToolPolicy` has these fields, in order:

```python
root: Path
allow_writes: bool = False
allowed_commands: tuple[tuple[str, ...], ...] = ()
max_file_bytes: int = 64 * 1024
max_write_bytes: int = 64 * 1024
max_list_entries: int = 200
max_tool_output_chars: int = 16_000
command_timeout_seconds: float = 30.0
```

`Workspace` exposes:

- `bind_receipt_store(store)` before the first dispatched action;
- `available_tools`, derived only from the policy;
- `resolve_path`, `list_files`, and `read_file` for bounded inspection;
- `write_file` and `edit_file` for the trusted, direct method layer;
- `run_command` for one exact allowed argument vector; and
- `execute(action, tool_call_id=...)` for the model-dispatch boundary.

Direct method calls make focused implementation tests possible. Model-requested
writes, edits, and commands go through `execute`, where they require an explicit
truthy `confirm_tool` response. Missing confirmation means No.

## Task 4: Bound Paths and Reads

Normalize the policy root once and remember its filesystem identity. Reject a
root that is missing, is a symlink, is the filesystem root or home directory, or
passes through protected metadata.

Tool paths are non-empty relative paths. Reject:

- `..` traversal and absolute paths;
- every symlinked path component;
- `.git` and common secret locations such as `.env`, `.ssh`, and `.aws`; and
- common private-key names and `.pem` or `.key` suffixes.

`list_files` returns at most `max_list_entries` sorted lines in `file path` or
`dir path` form. It omits symlinks and protected entries. `read_file` accepts
only a single-link regular UTF-8 file no larger than `max_file_bytes`, returns
its text, and remembers the digest of the bytes it inspected.

Expected failures include `AgentError` messages containing `path traversal`,
`symlinks`, `not accessible`, `not a regular file`, or `exceeds ... bytes`.

## Task 5: Inspect Before Overwrite

Creating a new file and replacing an existing file have different preconditions.
A new file may be prepared when writes are enabled. An existing file may be
overwritten only after `read_file` recorded its current digest.

Fail when the target bytes or mode change after that read. For `edit_file`, also
require non-empty old text that occurs exactly once. Write through a temporary
regular file in the same directory, flush it, atomically rename it, and flush
the parent directory. Preserve an existing file's permission bits; create new
files with mode `0600`.

Expected failures include `existing files must be read before overwrite`,
`file changed since it was read`, `file mode changed`, and `old text must match
exactly once`.

## Task 6: Put Approval Between Two Checks

`execute` first completes the write, edit, or command preflight without making a
change. Only then may it call `confirm_tool(action)`. If approval succeeds, the
workspace revalidates the prepared file and parent identity before committing.

This order handles two important cases:

1. an invalid action never reaches the approval callback; and
2. a file changed while the operator considered the request is not overwritten.

Recoverable dispatch failures return a string beginning with `error:` and are
still receipted. A missing or false callback returns `error: operator denied
<tool>` and performs no effect.

## Task 7: Allow Exact Commands and Enforce Timeouts

`allowed_commands` contains complete argument tuples. The requested list must
equal one tuple exactly; a prefix, suffix, different flag, or shell string is a
different command. Invoke the argument vector directly with the workspace as
its current directory—never with a shell.

Capture stdout and stderr together, cap the returned text, and preserve the
status line even when output is truncated. A successful result ends with
`[exit code: 0]`. A nonzero result begins or ends with an `error:` status. On a
timeout, kill the command's process group, reap the child, and return an error
that contains `command timed out after`.

Command approval does not make the executable safe. Keep the allowlist small and
run the course in a disposable workspace.

## Task 8: Make Receipts Immutable and Verifiable

`EffectReceipt` is a frozen dataclass with exactly these fields:

```python
tool_call_id: str
tool: str
arguments: dict[str, Any]
exit_state: str
result: str
changed_artifacts: tuple[str, ...]
```

Accept only the exit states `ok`, `error`, and `uncertain`. Detach and freeze the
JSON-compatible arguments so later caller mutation cannot change evidence.
Sort and deduplicate changed paths. `receipt_id` is the lowercase SHA-256 of the
canonical JSON payload; it does not include itself.

`to_dict()` adds `receipt_id` to a detached JSON representation.
`from_dict()` accepts exactly those durable fields, reconstructs the receipt,
and rejects a digest mismatch. Changing any stored argument, result, status, or
artifact must therefore fail closed.

`ReceiptStore(path)` loads an optional JSON-lines file and verifies every line.
`put` appends one canonical line, flushes it, and calls `fsync` before updating
the in-memory index. Repeating the same receipt is idempotent. Reusing a
`tool_call_id` for different evidence is an error. `get`, `require`, and
`by_tool_call` return only digest-verified receipts.

## Task 9: Receipt Every Dispatch

`Workspace.execute` assigns a non-empty call ID when none was supplied and
records the normalized inputs, bounded result, exit state, and exact file
changed by a successful write or edit. Bind a durable store before the first
dispatch when receipts must survive the process.

Validate a caller-supplied `tool_call_id` before any effect. Reject an ID already
present in the bound store instead of repeating its action. This checkpoint does
not guess what an interrupted process may have done: a timed-out command is
marked `uncertain`.

## Run the Cumulative Checkpoint

Run the learner workflow from the repository root:

```bash
pdm run test --week 4 --day 2
```

This copies the cumulative Day 2 learner test into `tests/` and runs it against
`tiny_llm`. The first tasks retain the Day 1 protocol checks; Tasks 4–9 exercise
the new workspace and receipt surface. During course development, the supplied
implementation can be checked without copying learner tests:

```bash
pdm run test-refsol --week 4 --day 2
```

The cumulative guards also compare public APIs, full signatures, dataclass
fields and defaults, package exports, and constants. Starter bodies must remain
TODO-only and cannot import the reference package or declare unpublished
capabilities.

{{#include copyright.md}}
