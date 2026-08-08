# Day 3: Safe Editing and Validation

> 🚧 **Early-review WIP:** This chapter is public for early review and may
> change. Use a disposable workspace when running the agent or enabling writes
> or commands.

Tools connect probabilistic model output to real side effects. Today you will
make that boundary explicit. File tools should remain inside one workspace,
mutations should require review, and successful work should be validated.

> **Implementation status:** The current Day 3 checkpoint covers bounded reads,
> protected roots and paths, rejection of detected symlinks, and writes disabled
> by default.
> Atomic writes, exact edits, and exact command allowlisting are checked on Day
> 5. Durable mutation intents, recovery, and default-No checkpoint undo belong
> to the cumulative Day 6 implementation, not this focused checkpoint.
> Automatic diff capture, validation-aware completion, and process isolation
> remain deferred.

## Current Checkpoint

Implement the read-side `ToolPolicy` and `Workspace` behavior in
`src/tiny_llm/agent/workspace.py`, then run:

```bash
pdm run test --week 4 --day 3
```

The expected checks list and read a normal file, hide a common secret, reject
unsafe paths and symlinks, and prove that writes are disabled by default. Use
`pdm run test-refsol --week 4 --day 3` for the supplied implementation.

## Workspace Boundaries

Resolve requested paths against a fixed workspace root. Reject filesystem root,
the user's home or one of its ancestors, and a root inside protected metadata or
secret paths. For tool paths, reject absolute paths, `..` components, and any
symbolic link detected during validation rather than following it. Deny access
to `.git` and common secret or key paths so the agent cannot recover hidden
benchmark solutions, rewrite repository metadata, or read obvious credentials.

Test at least these cases:

- a normal relative path;
- `../outside.txt`;
- an absolute path outside the workspace;
- a symlink inside the workspace pointing outside it; and
- a path whose parent does not exist yet.

Path checks protect `list_files`, `read_file`, `write_file`, and `edit_file`.
They do not confine `run_command`; keep the entire exercise workspace
disposable. They also do not form a hostile-filesystem sandbox: another process
can race pathname checks, and a hard link can expose bytes under a different
name.

## Approval Policy

Enabling writes or naming an exact command is a prerequisite, not blanket
permission. First complete the action schema checks and every tool-specific
preflight: path and content checks, observed-digest and unique-match checks for
file mutations, or exact argument-vector allowlisting for commands. Then pause
once before any side effect:

1. display the proposed `write_file`, `edit_file`, or `run_command` action;
2. ask a default-No `y/N` question;
3. continue only for an explicit `y` or `yes`;
4. for a file mutation, revalidate the path and observed digest before the
   atomic replacement; and
5. return a denial observation for empty input, EOF, non-interactive input, No,
   or any unrecognized response.

`list_files` and `read_file` do not prompt because their path and content limits
are enforced before access. This gate applies to model-dispatched actions through
`Workspace.execute()`; direct method calls form the trusted, model-free unit-test
layer. Session-wide approval is deliberately not inferred from one accepted
call. A denial is returned as the action's result so it remains visible to both
the model and the trace.

Confirmation does not make a command safe. `run_command` starts an exact
argument vector without a shell and with the workspace as `cwd`, but that program
can still read or modify any host path allowed to the process, spawn children,
or use the network. Strong confinement requires a container, virtual machine, or
equivalent sandbox.

## From a Stale-Write Guard to the Day 6 Mutation Journal

The recovery target performs these steps around an approved write or edit:

1. resolve and validate the destination;
2. read the current contents if the file exists;
3. record a content hash and before-image for later undo;
4. compute the proposed result in memory;
5. produce a diff for the trace or confirmation policy; and
6. replace the file atomically.

The Day 3 checkpoint records an in-memory content digest when it reads a file,
rejects byte changes detected during mutation preflight, rechecks the digest
after human approval, performs an atomic replace, and tracks paths changed by
file tools. It also retains an outcome-uncertain path if interruption or an
exception occurs between starting the commit and recording its completion, so
the CLI tells the operator to inspect that file. This catches ordinary changes
during the approval pause, but there is still a check-to-replace race against a
hostile concurrent filesystem actor. The digest is a stale-write guard, not a
durable journal, and checkpoint undo is not a Day 3 capability.

The cumulative reference implementation adds bounded before-images,
write-ahead mutation intents, restart classification, conflict checks, and
default-No checkpoint undo in Day 6. That later journal does not add automatic
diff capture or process isolation; follow the Day 6 chapter before claiming its
recovery and undo guarantees.

Do not automatically use `git reset`, `git checkout`, or a temporary commit as
the mutation journal. A learner may run the agent in a repository that already
contains unrelated work.

## Inspect Before Editing

A simple system instruction improves both reliability and reviewability:

```text
Inspect a file before editing it. Prefer exact edits over whole-file writes.
After changing code, run the smallest relevant validation command.
```

Policy code should still enforce what it can. For example, `edit_file` naturally
requires observed old text, an existing file must be read before `write_file`
can replace it, an observation digest must match at the post-approval recheck,
and both model-dispatched mutation tools require `y/N` approval.

## Validate After Editing

The final answer is not proof that the task is complete. Validation can include:

- a focused unit test;
- a type checker or linter;
- a formatter followed by a diff inspection; or
- a build command for the modified package.

Return command failures to the model with their exit status. The agent should
decide whether to inspect more code, make another edit, or report that it could
not finish within its budget.

The current loop does not know that a command was the task's required validation
and does not reject a final answer merely because the last command failed.
Validation-aware completion is a planned contract. Until it exists, inspect the
trace and exit codes rather than treating `completed=True` as proof of
correctness.

## Budgets and Repeated Failures

Add limits for:

- model turns;
- generated tokens;
- tool result bytes;
- command runtime; and
- consecutive invalid or identical actions.

If the same action repeats, whether it succeeds or fails, stop with a diagnostic
instead of spending the remaining budget on an obvious loop. Keep the raw
actions in the trace so repetition can become an evaluation metric.

## Cumulative Safety Exercise

Create a temporary fixture repository with:

- a small implementation file;
- a focused test exposing one bug;
- an unrelated file that must not change; and
- a symlink that points outside the workspace.

Ask the agent to fix the bug. The task passes only if:

1. the focused test succeeds;
2. the unrelated file is unchanged;
3. no **file tool** accesses or modifies a path outside the workspace; and
4. the trace contains an inspection before the first mutation.

If the trajectory uses `run_command`, allow only the exact fixture validation
command, approve it separately, and run the whole exercise in a disposable
environment. The baseline cannot prove that a subprocess avoided outside paths.

Run the same task with a deliberately malformed model action and a failing test
command. The agent should receive useful errors and remain within its budgets.

## Checkpoint

At the end of Day 3, you have the smallest genuinely useful coding agent: it can
inspect code, make a precise edit, and validate the result in a disposable
workspace after the cumulative Day 5 and Day 6 behavior is complete. The planned
remaining milestones make that agent persistent, efficient, controllable, and
measurable.

{{#include copyright.md}}
