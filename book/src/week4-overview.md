# 🚧 Week 4: Build a Coding Agent

> 🚧 **Course status:** The daily chapters are public for early review and may
> change. The repository contains tested checkpoints through Day 7, including
> safe static held-out evaluation. Executing model-authored code or held-out
> pytest remains deferred until an isolated container or virtual machine backend
> exists.

Weeks 1 through 3 turned tokens into text, made decoding efficient, and
introduced serving techniques. Week 4 adds the next layer: an agent loop that
lets the model observe a workspace, choose a tool, see the result, and continue
until a coding task is complete.

The goal is not to reproduce a production coding agent. It is to understand the
small mechanism underneath one and identify where reliability, efficiency, and
safety come from. By the end of the week, you will have a local CLI agent powered
by the inference stack you already built.

## What You Will Build

The Week 4 target agent can:

- inspect a repository without loading every file into the prompt;
- read bounded UTF-8 files and make exact, reviewable edits;
- run a narrowly scoped test command and use its output as feedback;
- continue an interactive conversation and resume it after process exit;
- reuse compatible KV-cache state instead of prefilling every turn from zero;
- compact an overlong context while retaining task state;
- checkpoint and undo its own file mutations;
- accept steering messages and interrupt long-running work; and
- solve a small inert repository task graded by held-out declarative checks.

This is a deliberately small target. Features such as multi-agent delegation,
remote execution, MCP integrations, and long-term user memory remain extensions
rather than prerequisites.

## Current Executable Baseline

The checked-in reference implementation is smaller than that target. It
currently provides:

- a stateless `generate_response()` adapter and a reusable `GenerationSession`;
- strict parsing of one final answer or one structured tool action;
- `list_files`, `read_file`, `write_file`, `edit_file`, and `run_command`;
- read-only workspace tools by default, with explicit tool enablement and
  per-dispatched-call approval;
- durable append-only session logs, resume, and project-instruction snapshots;
- rendered-token budgeting, bounded observations, and durable structured
  compaction;
- one cooperative cancellation signal across the loop, course-model decoding,
  file commits, and command polling;
- durable steering messages, write-ahead file-mutation intents, startup
  reconciliation, branch-local checkpoints, default-No undo plans, and session
  branches;
- sealed inert task packages, frozen candidate snapshots, deterministic static
  held-out checks, forbidden-path detection, and evaluation metrics; and
- a bounded loop that records actions, observations, confirmed and
  outcome-uncertain file-tool modifications, whether commands may have
  untracked or incompletely cleaned-up side effects, and its terminal reason.

`AgentRun.completed` records protocol completion—a valid final action from the
model—not task correctness. Day 7 keeps that field separate from the static
grader's `GradeReport` and `EvaluatedRun.task_success`. Checkpoint, undo, branch,
and steering support are programmatic APIs in Day 6, not interactive CLI
commands. The main CLI records an interrupted run and exits with status 130
after Ctrl-C.

## The Core Loop

Every chapter builds on the same loop:

1. Render the task, project instructions, recent events, and tool descriptions.
2. Decode one structured action using the model and KV cache.
3. Parse and validate the action before it reaches the operating system.
4. For a write, edit, or command, obtain a fresh default-No human approval.
5. Run the approved workspace tool and append its observation to the session.
6. Repeat until the model returns a final answer or reaches a budget.

The model does not edit files directly. It proposes an action; ordinary code
decides whether that action is valid and performs it. This boundary makes agent
behavior easy to inspect and test.

```text
task + session events
        |
        v
  context builder ---> model ---> action validator
        ^                            |
        |                            v
   tool result <--- tool runner <--- policy + conditional y/N gate
                         |
                         v
                      workspace
```

## Stateful Inference Extension

The starting model boundary is deliberately stateless:

```python
Generate = Callable[[list[Message]], str]
```

For every action, `generate_response()` renders the complete conversation,
creates a fresh cache for each model layer, prefills the entire prompt, decodes
one response, and releases the caches. That is a useful correctness baseline,
but an interactive agent repeatedly sends a long prompt whose prefix barely
changes.

Day 4 adds a callable
`GenerationSession` that keeps the same agent-facing API while owning token IDs
and layer caches. It compares the new rendered prompt with the cached token
sequence, rewinds a divergent suffix, and prefills only the new tokens. Days 5
and 6 reuse this operation after compaction, steering, and session branching.

The append-only event log remains the source of
truth. A process restart may rebuild KV state from events, so persisting K/V to
disk is an optional optimization rather than a correctness requirement.

The Qwen3-4B context budget is 32,768 total tokens. Day 5 starts
compaction at 24,576 input tokens and keeps the remaining 8,192 tokens for the
next response and tool output. This limit follows the model's training range,
not the amount of unified memory available; the derivation and long-context
measurements are in the
[performance appendix](./appendix-performance.md#long-context-budget-for-week-4).

## A Small Tool Surface

The baseline uses five tools inspired by small coding-agent harnesses. These
names and fields are the executable JSON protocol:

```text
list_files(path?)
read_file(path)
edit_file(path, old, new)
write_file(path, content)
run_command(argv)
```

`read_file` and `edit_file` are preferable to shell equivalents because they can
enforce consistent bounds and return structured errors. `list_files` provides
bounded discovery. `run_command` accepts a non-empty JSON array of exact
arguments, runs without a shell, and uses the timeout configured by the workspace
policy; the model cannot supply its own timeout.

The operator must first enable writes or name an exact allowed command. That is
only the outer policy gate. Once an eligible model-dispatched `write_file`,
`edit_file`, or `run_command` action passes schema, policy, and tool preflight,
the CLI asks for `y/N`; an empty response, EOF, non-interactive input, or any
answer other than an explicit yes denies the call. File mutations then revalidate
their path and observed digest before commit. A denial is returned to the model
as an observation.

A shell working directory is not a security sandbox. During this course, run the
agent only in a disposable exercise workspace. A production agent would need a
container, virtual machine, or similarly strong isolation boundary. An exact
allowed command can still delete files, read outside the workspace, launch child
processes, or use the network. Human confirmation reduces accidental execution;
it does not confine the approved program.

Day 7 therefore disables commands during evaluation and never imports or
executes candidate code. Its held-out checks parse bounded text, JSON, and
Python ASTs over a frozen snapshot. This is an honest static checkpoint, not a
general behavioral-code evaluator. Running candidate Python, pytest, compilers,
or task-local graders is deferred to a container or virtual machine backend.

Day 6 adds a write-ahead intent before each journaled file replace. On restart,
the journal compares the current content-and-mode fingerprint with the recorded
before and intended states and records `committed`, `not_applied`, or `conflict`
without changing the file. Checkpoint undo performs a whole-plan conflict
preflight and remains default-No. Safety copies retained from replacement or
undo are protected from tools and reported for manual inspection. Commands are
outside that reversible file journal and are never claimed as undoable.

The default demo now calls the course model through `GenerationSession`; the
`--solution mlx` compatibility backend retains the stateless adapter. The agent
loop, tools, and safety work remain the main arc, while interactive sessions
provide a focused inference-framework exercise without changing model kernels.

## Seven-Day Plan and Current Evidence

| Day | Planned topic | Current repository checkpoint |
| --- | --- | --- |
| 1 | Agent loop | Initial system and task messages are validated. |
| 2 | Tools and actions | Structured actions and enabled-tool prompts are validated. |
| 3 | Safety and validation | Bounded reads, protected paths, symlink rejection, and read-only defaults are validated. |
| 4 | Interactive sessions | Append-only sessions, resume checks, instruction snapshots, and live token-prefix reuse are validated. |
| 5 | Compaction | Exact rendered-token accounting, bounded observations, structured summaries, durable markers, and cache reconciliation are validated. |
| 6 | Control and recovery | Cooperative cancellation, durable interrupt and steering events, mutation reconciliation, checkpoints, conflict-safe undo, branches, and command cancellation with a fake process are validated. |
| 7 | Evaluation | Sealed inert packages, frozen snapshots, static held-out checks, forbidden-path grading, and metrics are validated without executing candidate code. |

Each daily chapter names the focused learner command for its current checkpoint.
For example:

```bash
pdm run test --week 4 --day 2
```

Use `pdm run test-refsol --week 4 --day 2` to check the supplied reference
implementation without copying the learner test.

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

The current command's workspace tools are read-only and expose no command runner
by default. Unless `--no-session` is selected, the CLI still writes its sensitive
local transcript under `.tiny-llm/sessions`.
`--allow-writes` and repeated `--allow-command "..."` flags make those tools
eligible. Each eligible action that passes preflight still defaults to **No** at
its `y/N` prompt. Keep `--root` pointed at a disposable exercise directory. Do
not interpret an exact command allowlist, a working directory, or a confirmation
prompt as process isolation.

The Day 4 CLI supports `--interactive`, `--continue`, `--session`, and
`--no-session`. Persistent transcripts are sensitive local files under
`.tiny-llm/sessions`; `--no-session` retains the earlier ephemeral behavior.
Ctrl-C during an agent run records an interruption when a durable session is in
use, reports known side effects, and exits with status 130. Day 6 checkpoint,
undo, branch, and live-steering operations are APIs for the exercise harness;
the CLI does not yet provide commands for them.

The Day 7 inspection CLI does not load a model or mutate a candidate. It can
validate package metadata and grade an unchanged fixture with the static grader:

```bash
pdm run evaluate-agent inspect evals/week4/localized-constant
pdm run evaluate-agent grade evals/week4/localized-constant
```

The grade command uses a fresh temporary stage and frozen snapshot. It keeps
commands disabled, supplies no automatic write approval, and never runs code
from the package. The shipped unchanged fixture is intentionally incorrect:
`inspect` exits zero, while `grade` prints a normal failed report and exits one.

The default Qwen3 4B model follows the structured action protocol more reliably.
Use `--model qwen3-0.6b` on memory-constrained machines and expect to spend more
time on malformed-action recovery.

## Target Milestones

- **Minimal:** the model can inspect a workspace and produce one valid action.
- **Useful:** the agent can make a precise change and run its test.
- **Recoverable:** the session can resume, compact, and undo its own changes.
- **Controllable:** budgets bind and the user can steer or interrupt work.
- **Efficient:** compatible turns reuse their unchanged prompt prefix.
- **Measurable:** a repeatable task suite distinguishes progress from anecdotes.

## Further Reading

- [Pi coding agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)
- [Benchmarking Coding Agents on Databricks' Multi-Million Line Codebase](https://www.databricks.com/blog/benchmarking-coding-agents-databricks-multi-million-line-codebase)

{{#include copyright.md}}
