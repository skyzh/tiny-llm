# 🚧 Day 3: Edit, Validate, and Record

Day 2 gave the model two read-only tools. It could inspect a disposable project
and explain what it found, but it could not fix anything. Day 3 completes one
small coding cycle:

```text
read file -> propose exact edit -> operator approves -> recheck bytes
    -> replace file -> record receipt -> run focused check -> record receipt
    -> final answer
```

The important idea is not broad autonomy. It is an explicit boundary between a
model proposal and a local side effect.

Before running the focused cycle, predict three facts: how many approvals and
receipts exist after a denied effect, whether stale-read detection changes any
file bytes, and which evidence remains after a validation command returns a
nonzero status. The workspace, receipt log, and event trace let you check those
answers independently of the model's final sentence.

## The Teaching Boundary

Use this checkpoint with one trusted operator, one Python process, and a
disposable repository that contains no secrets. Its path checks prevent ordinary
mistakes, but they are not a sandbox or a defense against a hostile filesystem.
The command tool is not a process jail: an allowed program can access anything
the host process can access, spawn children, or use the network.

The receipt file is a simple append-only JSONL teaching record. It detects
edited receipt bytes when reopened and handles a repeated call ID in the same
process. It is not a transaction log, an `fsync` protocol, a multi-writer store,
or proof that an interrupted effect did or did not happen. Day 3 deliberately
stops at this receipt boundary.

## Files and Public Surface

Later checkpoint, compaction, steering, evaluation, branching, and evidence
modules are already declared in the cumulative Day 9 scaffold. Their later-day
implementation work is out of scope. Day 3 owns only the following Day 1--3
surfaces:

Implement the TODO bodies in these starter files:

| File | Public names | Responsibility |
| --- | --- | --- |
| `src/tiny_llm/agent/workspace.py` | `ToolPolicy`, `Workspace` | Authorize reads, approved edits, and one exact validation command. |
| `src/tiny_llm/agent/receipts.py` | `EffectReceipt`, `ReceiptStore` | Represent effects and optionally append verified JSONL records. |
| `src/tiny_llm/agent/__init__.py` | Day 1--3 names within the final scaffold | Export the two receipt types. |

`ToolPolicy` keeps its first three Day 2 fields and adds:

```python
allow_writes: bool = False
allowed_commands: tuple[tuple[str, ...], ...] = ()
max_write_bytes: int = 64 * 1024
command_timeout_seconds: float = 30.0
```

Writes stay disabled unless `allow_writes=True`. Commands stay disabled unless
their complete argument tuple appears in `allowed_commands`. There is no shell
string or prefix match.

`Workspace(policy, confirm_tool=None)` creates an in-memory receipt store. Pass
a `ReceiptStore(path)` as the third argument when you want JSONL output. The
workspace extends the Day 2 methods with `write_file`, `edit_file`,
`run_command`, and `modified_files`. `execute(action, tool_call_id=None)` is the
model-facing gate. Direct tool methods are useful for focused unit tests;
`execute` performs the approval and receipt steps.

## Task 1: Authorize Tools Explicitly

Build `available_tools` from the policy. Listing and reading are always present.
Add both file mutation tools only when writes are enabled, and add
`run_command` only when at least one exact command is configured. Validate all
size limits, the timeout, the boolean flag, and every command part.

This configuration permits one focused check:

```python
validation = ("python", "-m", "pytest", "tests/test_math.py", "-q")
policy = ToolPolicy(
    Path("demo-project"),
    allow_writes=True,
    allowed_commands=(validation,),
)
```

## Task 2: Read Before Changing Existing Bytes

Keep Day 2's path and read rules. When `read_file` succeeds, remember the
SHA-256 digest of the bytes that were returned. Replacing or editing an existing
file requires that observation. A new file does not have old bytes to inspect,
but its parent directory must already exist.

For `edit_file`, require a non-empty `old` string that occurs exactly once.
Compute the proposed bytes in memory and enforce `max_write_bytes` before asking
for approval. Whole-file `write_file` replacements also require a prior read.

## Task 3: Ask Once, Default No, Then Recheck

`execute` preflights the complete action before calling `confirm_tool`. Missing
callbacks, `False`, and every value other than the boolean `True` deny the
effect. A terminal program can provide a small default-No callback:

```python
def confirm(action):
    answer = input(f"Approve {action.tool} {action.arguments}? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}
```

After approval, `write_file` or `edit_file` reads the destination again and
compares its digest with the earlier observation. If another actor changed the
bytes while the operator was deciding, return `error: file changed since it was
read` and do not overwrite them.

## Task 4: Replace Through the Same Directory

Write the proposed bytes to a temporary file in the destination's parent, close
it, and call `os.replace(temporary, destination)`. Clean up a leftover temporary
file after an error. This avoids presenting a partially written destination to
ordinary readers.

This small pattern is atomic at the replacement step, but it is not a durable
journal and does not close the check-to-replace race against a hostile actor.

## Task 5: Run One Exact Validation Command

`run_command(argv)` accepts a non-empty list of strings only when its tuple is
exactly allowlisted. Call `subprocess.run` without a shell, with the workspace
root as `cwd`, captured text output, and the configured timeout. Bound combined
stdout and stderr so one observation cannot consume the whole context window.

Return one observation with the status and captured output:

```text
status: 0
output:
1 passed
```

A nonzero status and a timeout are ordinary validation results the model can
inspect. They are not Python exceptions and do not prove the final answer is
correct.

## Task 6: Record Simple Effect Receipts

An `EffectReceipt` has these fields, in order:

```python
tool_call_id: str
tool: str
arguments: dict[str, Any]
exit_state: str
result: str
changed_artifacts: tuple[str, ...] = ()
```

Its `receipt_id` is the SHA-256 digest of the canonical JSON payload. A
successful write or edit records exactly one normalized workspace-relative
artifact. A validation receipt records the exact `argv`, status and captured
output, with no changed artifacts. `ReceiptStore(path)` loads and verifies an
existing JSONL file; `ReceiptStore()` remains in memory.

The store maps one `tool_call_id` to one receipt. Repeating the same call ID and
action returns the existing result without running the effect again. Reusing the
ID for another action is an error. This is proportional duplicate handling for
one process, not distributed exactly-once execution.

## Task 7: Run the Complete Scripted Cycle

The test uses scripted model responses, so it needs no model weights:

```python
responses = iter([
    '{"tool":"read_file","path":"app.py"}',
    '{"tool":"edit_file","path":"app.py","old":"1","new":"2"}',
    '{"tool":"run_command","argv":["python","-m","pytest","tests/test_math.py","-q"]}',
    '{"final":"changed and validated app.py"}',
])

store = ReceiptStore(Path("demo-project/.agent-receipts.jsonl"))
workspace = Workspace(policy, confirm, store)
result = run_agent("fix app.py", lambda _messages: next(responses), workspace)
```

Inspect `result.events`, `workspace.modified_files`, and the two receipts. The
edit receipt names `app.py`; the validation receipt has an empty artifact tuple.
The final answer is still a model statement, so the validation status in the
trace is the evidence that matters.

## Run the Day 3 Checkpoint

From the repository root, copy and run the learner checkpoint explicitly:

```bash
pdm run copy-test --week 4 --day 3
pdm run test --week 4 --day 3
```

Before you implement the TODOs, the copied test is expected to fail because the
new starter methods return `None`. `copy-test` refreshes the learner copy from
the supplied checkpoint. Keep those failures until you solve each task; do not
import `tiny_llm_ref` from the starter. This is a day-local test, not a
cumulative Days 1--3 runner.

Course maintainers can check the supplied implementation without copying the
learner test:

```bash
pdm run test-refsol --week 4 --day 3
```

The course-code guard checks exact public signatures, dataclass fields,
package exports, and TODO-only starter bodies across the final declaration
scaffold.

## Explore the Full Cycle with a Real Model

Keep the scripted checkpoint as the deterministic proof. This manual exercise
lets a real model plan the same Day 3 cycle from one natural-language goal.
Its wording and tool order can vary, and completion is not an automated test.
Use a fresh disposable directory with no secrets.

The CLI default is `qwen3-4b` (`Qwen/Qwen3-4B-MLX-4bit`), whose cached weights
use about 2 GiB; use that lower-resource option when needed. The recorded
exploratory run below used `mlx-community/Qwen3-30B-A3B-4bit`, whose cached
weights use about 16 GiB and require sufficient Apple unified memory. Model
behavior and tool order vary with either choice. Both require macOS on Apple
Silicon and the installed MLX dependencies. An uncached first run downloads
the selected weights from Hugging Face and needs network access plus the
corresponding free disk space. If MLX, network access, disk space, unified
memory, or the weights are unavailable, model loading fails before any tool
call; do not treat a scripted checkpoint as evidence that this live run
occurred.

As on Day 2, this CLI's supplied MLX-LM adapter bypasses the learner-owned
`generate_response()` helper. It does exercise your loop, workspace,
approval, effect, and receipt paths.

Pre-create the workspace, an existing file, and one focused validation fixture:

```bash
EDIT_ROOT="$(mktemp -d)"
printf '%s\n' \
  'def greeting(name):' \
  '    return f"Hello, {name}!"' > "$EDIT_ROOT/app.py"
cat > "$EDIT_ROOT/validate.py" <<'PY'
from pathlib import Path
from app import greeting

assert greeting("Ada") == "Welcome, Ada!"
assert Path("NOTES.md").read_text() == "Greeting now says Welcome.\n"
print("validation passed")
PY
```

Now give the real model one goal. `--allow-command` names the only command it
may run, including its exact arguments. Each proposed write, edit, or command
still pauses at a default-No learner approval prompt:

```bash
pdm run agent -- --model mlx-community/Qwen3-30B-A3B-4bit --root "$EDIT_ROOT" \
  --allow-writes \
  --allow-command "python validate.py" \
  --receipt-log .agent-receipts.jsonl \
  "Inspect the workspace. Create NOTES.md containing exactly 'Greeting now says Welcome.' followed by a newline, precisely change app.py so greeting says Welcome instead of Hello, run the allowed validation, react to its evidence, and finish with a brief summary. Use workspace tools to gather evidence before any effect. Your first response must be one tool request, every response must contain exactly one JSON object, and do not finish until the requested files and validation evidence have been inspected."
```

Read every approval payload before answering `y`. The live trace shows the
model response, parsed action, tool observation, validation status/output, and
final answer. Afterward, inspect the changed bytes and the durable effect
receipts rather than trusting the final prose alone:

```bash
cat "$EDIT_ROOT/app.py" "$EDIT_ROOT/NOTES.md"
cat "$EDIT_ROOT/.agent-receipts.jsonl"
```

The receipt records preserve the approved `write_file`, `edit_file`, and
`run_command` arguments and outcomes. The status and captured output in the
validation receipt are the evidence to compare with the changed bytes.

## Checkpoint

You now have a small end-to-end coding loop: inspect real bytes, propose one
precise change, pause for a trusted operator, reject stale observations, replace
the file, validate with one exact command, and retain simple evidence of both
effects. Continue with [Day 4: Checkpoint and Resume](week4-04-sessions.md) to
save one complete observation boundary and restore it through a fresh scripted
model without turning Day 3 into production infrastructure.

Day 3 is currently the last mechanism exposed by the real-model CLI. Days
4--9 use deterministic library checkpoints until the supplied Week 4 capstone
connects them into one runnable product path.

{{#include copyright.md}}
