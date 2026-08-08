# Day 7: Safe Deterministic Evaluation

> 🚧 **Early-review WIP:** This chapter is public for early review and may
> change. Use a disposable workspace when running the agent or enabling writes
> or commands.

An agent saying that it finished is not evidence that its change is correct.
Day 7 adds a small, deterministic evaluation harness that grades the resulting
files independently of the model's final message. The checkpoint is deliberately
static: it can inspect text, JSON, and Python syntax, but it never imports or
executes candidate code.

> **Implementation status:** The reference implementation, learner API surface,
> inert task packages, and focused tests implement the static evaluation boundary
> described here. The chapter remains WIP even though the checkpoint is
> executable. Executable Python tests and general coding-task grading remain
> deferred until the candidate can run inside a container or virtual machine.

## Check the Chapter

Implement the evaluation APIs under `src/tiny_llm/agent/`, then run:

```bash
pdm run test --week 4 --day 7
```

Use `pdm run test-refsol --week 4 --day 7` for the supplied implementation. The
focused suite uses scripted model actions and inert fixtures. Tripwires verify
that grading does not launch a process, import a candidate module, or call
`eval()` or `exec()`.

You can inspect and grade an unchanged task fixture without loading a model:

```bash
pdm run evaluate-agent inspect evals/week4/localized-constant
pdm run evaluate-agent grade evals/week4/localized-constant
```

The second command stages the fixture into a fresh temporary directory, freezes
that candidate, and runs only the declarative held-out checks. It does not run
the coding agent or approve a mutation. Use the Python API to evaluate a scripted
or model-backed trajectory. `inspect` exits zero for a valid package; `grade`
exits zero for a pass, one for a deterministic failure, and two for an invalid
package or CLI error. The shipped unchanged localized-constant baseline is
intentionally incorrect, so grading it returns a normal failed report.

## A Sealed, Inert Task Package

Each task is a directory with one public manifest, a starting workspace, and a
held-out declarative check file:

```text
localized-constant/
  task.json
  workspace/
    answer.py
  held_out_tests/
    checks.json
```

`task.json` is strict and versioned:

```json
{
  "schema_version": 1,
  "id": "localized-constant",
  "prompt": "Correct the answer without changing the public name.",
  "max_steps": 6,
  "editable_paths": ["answer.py"]
}
```

The loader rejects missing and unknown fields, unsupported schema versions,
blank identifiers or prompts, invalid budgets, absolute paths, parent traversal,
duplicate editable paths, and overlapping editable path prefixes. It also
rejects symlinks, `.git`, excessive file counts or bytes, unknown package
entries, and executable held-out files. A package cannot contain `grade.py`, a
command, or another task-local program.

`TaskPackage.stage()` copies only `workspace/` into a new narrow destination.
The destination must not already contain a candidate tree. Held-out checks,
solution history, and package metadata never enter the agent-visible workspace.
Staging records a sorted snapshot of every initial regular file and a canonical
tree SHA-256 that also covers directory presence, without modifying the original
package.

These rules make the included fixtures inert and reproducible. They are not a
general way to make an untrusted repository safe. Parsing a package still reads
bytes supplied by its author, and the later agent run can only be as isolated as
its enabled tools.

## Keep Commands Disabled and Writes Default-No

`evaluate_task()` creates a `Workspace` whose command allowlist is empty. An
agent action requesting `run_command` is rejected before process launch. The
evaluation harness does not translate manifest data into an argv and does not
offer a task-local validation command.

File writes remain subject to the normal workspace checks and a per-call human
confirmation callback. If no callback is supplied, the write is denied. The
evaluator never installs an always-yes callback on the caller's behalf. A CLI or
application that wants model-authored mutations must present the exact tool call
and implement the same explicit `y/N` policy as the main agent CLI: blank input,
EOF, non-interactive input, interruption, and every response except `y` or `yes`
mean No.

This gives the evaluator a small capability boundary:

```text
public manifest + staged workspace
                |
                v
       bounded agent loop
       writes: explicit y/N
       commands: disabled
                |
                v
       frozen candidate tree
                |
                v
     static held-out grader
```

It does not make host execution safe. Do not enable `run_command` to make a task
more realistic: an approved Python program or test can delete files outside the
workspace, read credentials, access the network, or create detached processes.

## Freeze Before Revealing Held-Out Checks

The evaluation lifecycle is ordered so hidden expectations cannot influence the
agent trajectory:

1. Validate the package and stage only its public workspace.
2. Record the initial file snapshots and tree hash.
3. Run the bounded agent with commands disabled and writes defaulting to No.
4. Stop after a valid final action, budget termination, interruption, or another
   recorded terminal reason.
5. Copy the candidate into a separate, read-only-in-practice grading snapshot.
6. Record `evaluation_started` with the task ID and candidate tree hash.
7. Load `held_out_tests/checks.json` for the first time.
8. Run the allowlisted static checks over the frozen snapshot.
9. Record `evaluation_result` with the grade, ordered check outcomes, forbidden
   modifications, and metrics.

The grader never observes a live tree that the agent can continue changing.
This checkpoint treats one staged task and its session as single-use. A crash
after `evaluation_started` but before `evaluation_result` leaves an incomplete
record; restage and rerun the task instead of silently grading different bytes.
Hidden expected values are not included in model messages or pre-grade session
events.

Freezing closes an evaluator consistency gap, not every host race. A separate
same-user process can still interfere with ordinary files while they are copied.
Use a container or virtual machine when task inputs or neighboring processes are
untrusted.

## Declarative Held-Out Checks

`StaticHeldOutGrader` accepts a versioned JSON document containing only a small
allowlist of checks:

- `path_exists` requires a regular file at a relative path;
- `path_absent` requires that a relative path not exist;
- `text_equals` compares bounded UTF-8 content;
- `json_value` parses JSON and compares a value at a declared path;
- `python_constant` parses Python with `ast.parse()` and inspects one literal
  assignment; and
- `unchanged` compares a path with its initial snapshot.

Every check has strict fields and bounded inputs. Unsupported or malformed
checks produce an evaluation error, never a false pass. Results and paths are
sorted so the same manifest, check specification, and candidate bytes produce
the same `GradeReport`.

The Python check is syntax inspection, not program execution. It does not import
the candidate, add the staged directory to `sys.path`, evaluate an expression,
execute a module, or start a child process. A file may contain dangerous-looking
top-level calls; `ast.parse()` represents them as nodes and does not run them.
The check accepts only the supported literal assignment shape and rejects other
expressions.

After the declared checks, the grader compares the frozen tree with the initial
snapshot. Creating, changing, or deleting a path outside `editable_paths` is a
forbidden modification and fails the grade even when every held-out check passes.
The candidate tree hash and ordered forbidden-path list make that decision
auditable.

Static checks intentionally cover less behavior than executable tests. They are
well suited to localized constants, exact text, repository configuration, file
presence, and small syntax-shape exercises. They cannot prove that arbitrary
Python code behaves correctly.

## Completion Is Not Correctness

The loop and grader answer different questions:

| Signal | Meaning |
| --- | --- |
| `AgentRun.completed` | the model returned a valid final action |
| `AgentRun.reason` | why the bounded agent loop stopped |
| `GradeReport.status` | whether deterministic checks passed, failed, or errored |
| `EvaluatedRun.task_success` | whether `GradeReport.status == "passed"` |

A fluent final message paired with wrong bytes is completed but failed. A run
that exhausts its step budget can still pass when its final frozen bytes satisfy
every check. A grader configuration error is distinct from an ordinary failed
check. Model prose, timestamps, latency, and formatting never decide the grade.

Keep these fields separate in reports and dashboards. Conflating protocol
completion with task success rewards agents for stopping confidently rather
than for producing correct files.

## Metrics From Durable Events

`aggregate_metrics()` derives evaluation metrics from the retained session
events instead of mutable counters in the runner. It reports:

- model turns, tool calls, malformed actions, and tool errors;
- input and output tokens when the generation backend reports them;
- reused, rewound, and newly prefetched tokens;
- visible tool-result bytes and compaction count;
- generation latency when available, wall-clock evaluation time, and terminal
  reason.

A scripted string-only generator has no token or cache measurements, so those
fields remain unknown rather than becoming misleading zeroes. Grade ordering and
status do not depend on performance metrics.

The complete session trace is retained to explain a score and compare
trajectories. It can contain prompts, model output, source excerpts, tool
results, diffs, before-images, and local paths. Treat the session and evaluation
records as sensitive data: do not publish them merely because the aggregate
metrics are safe to share.

## Exercises

1. Inspect an inert package and confirm that held-out expected values are not
   printed.
2. Grade its unchanged baseline twice and compare the ordered report and tree
   hash.
3. Run a scripted trajectory that makes the expected edit and compare
   `AgentRun.completed` with `EvaluatedRun.task_success`.
4. Return a valid final action without editing the file and observe a completed,
   failed evaluation.
5. Try to modify a path outside `editable_paths` and inspect the forbidden-path
   result.
6. Put a dangerous-looking call at Python module scope and verify that
   `python_constant` parses but never executes it.
7. Request `run_command` from the scripted agent and verify that no process is
   launched.
8. Deny a proposed edit with blank input and verify that grading uses the
   unchanged frozen candidate.

## What Remains Deferred

This checkpoint does **not** execute model-authored Python, pytest, compilers,
build systems, task-local graders, or manifest-provided commands. A temporary
directory and `cwd` are path-selection conveniences, not confinement. Running
those programs on the host would give candidate code the evaluator's filesystem,
process, credential, and network authority.

General behavioral coding evaluation therefore requires a later backend with a
real isolation boundary, such as a disposable container or virtual machine with
a read-only base image, an explicit writable mount, network disabled by default,
resource limits, process cleanup, and trusted tests supplied from outside the
candidate workspace. Only that backend should install and run held-out pytest or
attempt the broader parser capstone.

Other useful extensions include repeated sampled runs, model comparisons,
statistical confidence intervals, leak-resistant external task distribution,
and durable restart of an in-progress grading job. None of them changes the Day
7 rule: task correctness comes from an independent, deterministic grader, and
executable candidate code must not run without actual isolation.

{{#include copyright.md}}
