# Day 3: Safe Editing and Validation

Tools connect probabilistic model output to real side effects. Today you will
make that boundary explicit: every path is confined, every mutation is
reviewable, and successful work is validated.

## Workspace Boundaries

Resolve requested paths against a fixed workspace root. A valid path must remain
under that root after resolving `..` components and symbolic links. Deny access
to `.git` so the agent cannot recover hidden benchmark solutions or rewrite
repository metadata.

Test at least these cases:

- a normal relative path;
- `../outside.txt`;
- an absolute path outside the workspace;
- a symlink inside the workspace pointing outside it; and
- a path whose parent does not exist yet.

Path checks protect file tools. They do not confine arbitrary shell commands;
keep the entire exercise workspace disposable.

## Mutation Policy

Before a write or edit:

1. resolve and validate the destination;
2. read the current contents if the file exists;
3. record a content hash and before-image for later undo;
4. compute the proposed result in memory;
5. produce a diff for the trace or confirmation policy; and
6. replace the file atomically.

Do not automatically use `git reset`, `git checkout`, or a temporary commit as
the mutation journal. A learner may run the agent in a repository that already
contains unrelated work.

## Inspect Before Editing

A simple system instruction improves both reliability and reviewability:

```text
Inspect a file before editing it. Prefer exact edits over whole-file writes.
After changing code, run the smallest relevant validation command.
```

Policy code should still enforce what it can. For example, `edit` naturally
requires observed old text, and `write` can warn or require confirmation before
overwriting an existing file.

## Validate After Editing

The final answer is not proof that the task is complete. Validation can include:

- a focused unit test;
- a type checker or linter;
- a formatter followed by a diff inspection; or
- a build command for the modified package.

Return command failures to the model with their exit status. The agent should
decide whether to inspect more code, make another edit, or report that it could
not finish within its budget.

## Budgets and Repeated Failures

Add limits for:

- model turns;
- generated tokens;
- tool result bytes;
- command runtime; and
- consecutive invalid or identical actions.

If the same action fails repeatedly, stop with a diagnostic instead of spending
the remaining budget on an obvious loop. Keep the raw actions in the trace so
the failure can become an evaluation metric.

## Exercise

Create a temporary fixture repository with:

- a small implementation file;
- a focused test exposing one bug;
- an unrelated file that must not change; and
- a symlink that points outside the workspace.

Ask the agent to fix the bug. The task passes only if:

1. the focused test succeeds;
2. the unrelated file is unchanged;
3. no file outside the workspace is accessed or modified; and
4. the trace contains an inspection before the first mutation.

Run the same task with a deliberately malformed model action and a failing test
command. The agent should receive useful errors and remain within its budgets.

## Checkpoint

At the end of Day 3, you have the smallest genuinely useful coding agent: it can
inspect code, make a precise edit, and validate the result in a disposable
workspace. The remaining days make that agent persistent, efficient,
controllable, and measurable.

{{#include ../copyright.md}}
