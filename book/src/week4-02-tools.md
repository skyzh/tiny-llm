# 🚧 Day 2: Inspect a Workspace

Day 1 built a loop around a fake workspace. The model could request a tool,
the loop could return an observation, and the model could finish—but no tool
looked at a real project.

Day 2 adds the smallest useful workspace: two read-only tools that let the
model list a directory and read a text file. This makes the complete cycle
visible without mixing in approval, edits, commands, or durable receipts.

```text
model requests list_files
          |
          v
 Workspace checks the path
          |
          v
 loop returns the listing
          |
          v
 model requests read_file
          |
          v
 loop returns the file text
          |
          v
 model returns a final answer
```

## The Teaching Boundary

Use these tools with a trusted operator and a disposable local repository.
`Path.resolve()` containment and symlink refusal prevent ordinary path mistakes,
but they are not a sandbox or a defense against a hostile filesystem. The tool
result also becomes model input, so do not point the workspace at files you
would not send to the model.

[Day 3](week4-03-safe-editing.md) adds operator-approved edits, one allowlisted
validation command, and a simple receipt log. Keeping those effects out of Day
2 lets you first see the read-only agent loop clearly.

## Files and Public Surface

Implement the TODO bodies in:

| File | Public names | Responsibility |
| --- | --- | --- |
| `src/tiny_llm/agent/workspace.py` | `ToolPolicy`, `Workspace` | Bound one directory and expose `list_files` plus `read_file`. |
| `src/tiny_llm/agent/__init__.py` | Day 1 API plus `ToolPolicy`, `Workspace` | Make the cumulative checkpoint importable. |

`ToolPolicy` has three fields, in order:

```python
root: Path
max_file_bytes: int = 64 * 1024
max_list_entries: int = 200
```

`Workspace` exposes:

- `available_tools`, always `{"list_files", "read_file"}`;
- `resolve_path(raw, must_exist=True)`;
- `list_files(raw=".")`;
- `read_file(raw)`; and
- `execute(action)`.

The starter declarations are the contract. Do not add write, command, approval,
receipt, session, checkpoint, or rewind APIs yet.

## Task 1: Normalize One Workspace Root

`ToolPolicy` receives the directory the agent may inspect. Convert it to a
resolved `Path`, require an existing directory, reject a symlink root, and
require both numeric limits to be positive.

Normalizing once keeps every later check relative to the same root:

```python
policy = ToolPolicy(Path("demo-project"))
workspace = Workspace(policy)
```

## Task 2: Resolve Ordinary Relative Paths

`resolve_path()` accepts a non-empty relative path and returns its resolved
location under `policy.root`. Reject:

- absolute paths and `..` traversal;
- symlinked path components;
- `.git`, `.env`, `.ssh`, `.aws`, and common credential or private-key names;
- missing paths when `must_exist=True`; and
- any resolved path outside the workspace.

Return a recoverable `AgentError` for these cases. The agent loop can turn that
error into an observation instead of crashing.

This is deliberately an ordinary local-repository boundary. A different
process can still race filesystem checks; defending against a hostile
filesystem is outside this course checkpoint.

## Task 3: List One Directory

`list_files()` lists direct children in sorted order. Emit one line per visible
regular file or directory:

```text
file README.md
dir src
```

Omit protected names, symlinks, and special files. Stop after
`max_list_entries` lines. Return `(empty directory)` when no visible entry
remains.

The tool is intentionally not recursive. The model can request another
`list_files` action for a directory it wants to inspect.

## Task 4: Read One Text File

`read_file()` accepts a visible regular file no larger than `max_file_bytes`.
Read its bytes and decode UTF-8. Reject directories, oversized files, invalid
UTF-8, protected paths, and symlinks with `AgentError`.

The size bound keeps a single observation from consuming the whole context
window. Later checkpoints can add more deliberate context selection; Day 2
only needs one obvious limit.

## Task 5: Turn Failures into Observations

`execute()` dispatches a parsed `ToolAction` to `list_files` or `read_file`.
It returns successful text directly. A recoverable failure becomes a string
beginning with `error:` so `run_agent()` can append it to the conversation and
let the model choose another action.

A known future tool such as `write_file` is still disabled on Day 2. The system
prompt describes only `workspace.available_tools`, and `parse_action()` checks
that enabled set before dispatch.

## Task 6: Run the Read-Only Cycle

The checkpoint uses a scripted model so it is deterministic and does not load
weights:

```python
responses = iter([
    '{"tool":"list_files"}',
    '{"tool":"read_file","path":"README.md"}',
    '{"final":"README says hello"}',
])

result = run_agent(
    "inspect the project",
    lambda messages: next(responses),
    workspace,
)
```

Inspect `result.events`: the first two events contain the parsed tool actions
and exact observations, and the third contains the final answer. The same tool
result appears in the next model input as `Tool result:\n...`.

## Run the Cumulative Checkpoint

From the repository root, run:

```bash
pdm run test --week 4 --day 2
```

This copies the cumulative Day 2 learner test into `tests/` and runs it against
`tiny_llm`. During course development, check the supplied implementation
without copying the learner test:

```bash
pdm run test-refsol --week 4 --day 2
```

The cumulative course-code guard compares the starter and reference public
signatures, dataclass fields, package exports, and solution-free method bodies.

## Explore with a Real Model

The scripted checkpoint above is the reproducible mechanics proof. This
separate manual run is exploratory: a real model chooses the tools, so its
wording and exact tool order may vary. Use only a disposable directory that
contains no secrets or private source.

The CLI defaults to `qwen3-4b` (`Qwen/Qwen3-4B-MLX-4bit`). The command below
selects the stronger `mlx-community/Qwen3-30B-A3B-4bit` model for more reliable
tool use. Both use the local MLX model path already used by this repository.
You need macOS on Apple Silicon and the installed MLX dependencies. The first
run downloads the selected weights from Hugging Face when they are not cached,
so it also needs network access and several gigabytes of free disk. If MLX is
unavailable or the weights cannot be loaded, the command exits before the
agent calls a workspace tool; it does not substitute scripted output.

Create a tiny read-only workspace, then give the model one goal:

```bash
INSPECT_ROOT="$(mktemp -d)"
mkdir "$INSPECT_ROOT/src"
printf '%s\n' '# Pocket Weather' 'A tiny terminal forecast project.' > "$INSPECT_ROOT/README.md"
printf '%s\n' 'def forecast(city):' '    return f"Sunny in {city}"' > "$INSPECT_ROOT/src/weather.py"

pdm run agent -- --model mlx-community/Qwen3-30B-A3B-4bit --root "$INSPECT_ROOT" \
  "Inspect this workspace and explain its purpose and the behavior implemented in its source file. Use the available workspace tools to gather evidence from both the project overview and the source file. Your first response must be one tool request, every response must contain exactly one JSON object, and you must not finish until you have read the source file."
```

Watch the printed goal, model responses, parsed actions, and tool observations.
There is no predefined action list: the model decides what to list and read
before returning a final answer.
The policy is read-only, so no approval prompt or receipt is expected. Compare
the final answer with the actual disposable files:

```bash
find "$INSPECT_ROOT" -type f -print
cat "$INSPECT_ROOT/README.md" "$INSPECT_ROOT/src/weather.py"
```

When this checkpoint is green, continue to [Day 3: Edit, Validate, and
Record](week4-03-safe-editing.md).

{{#include copyright.md}}
