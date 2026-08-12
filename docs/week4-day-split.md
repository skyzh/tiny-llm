# Week 4 Day Split (reference for reviewers)

Status: Days 1--4 are published checkpoints. Days 2--4 follow Chi's approved
simplified agent-loop cycle, with each day shipping as one cumulative learner
PR so reviewers can see exactly what belongs to that checkpoint.

## 7-day structure

| Day | Theme | Features (PRs) | Modules |
|---|---|---|---|
| 1 | Validated agent loop + tool protocol | feat1 | `protocol.py`, `loop.py` |
| 2 | Inspect a workspace | read-only list/read tools | `workspace.py` |
| 3 | Edit, validate, and record | approved edits, one command, simple receipts | `workspace.py`, `receipts.py` |
| 4 | Checkpoint and resume | one conversation + fake-model cache snapshot | `checkpoint.py`, `loop.py` |
| 5 | Reserved learner checkpoint | unpublished | — |
| 6 | Reserved learner checkpoint | unpublished | — |
| 7 | Reserved learner checkpoint | unpublished | — |

Extension (not a day): COW/radix cache — `docs/week4-cow-radix-extension-plan.md`.

## Delivery shape

- Each learner day is one complete Draft PR based on the previously merged day.
- The PR includes the reference, solution-free starter, focused course-code
  tests, and learner chapter for that checkpoint.
- Days are implemented sequentially. A later day does not leak API or prose
  into the current starter.

## Why 7 days

The existing `week4-01..07` chapter numbering stays stable. Each remaining day
adds one visible agent-loop concept; scaling and production-hardening machinery
stay outside the core course unless a later checkpoint explicitly teaches it.
