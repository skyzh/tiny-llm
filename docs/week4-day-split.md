# Week 4 Day Split (reference for reviewers)

Status: decided by Forge per Chi's instruction ("7/10/x days your call"); the
refsol stack implements features one PR per day so reviewers can see exactly
what belongs to each day. Curriculum prose (book) is not part of this stack.

## 7-day structure

| Day | Theme | Features (PRs) | Modules |
|---|---|---|---|
| 1 | Validated agent loop + tool protocol | feat1 | `protocol.py`, `loop.py`, `generation.py` (minimal) |
| 2 | Authorize effects + durable receipts | feat2 | `workspace.py` (simplified), `receipts.py` |
| 3 | Session tree (event-level id/parentId) | feat3 | `session.py` |
| 4 | KV checkpoint/resume + sequential rewind | feat4+5 | `checkpoint.py`, `branch.py`, `generation.py` (session) |
| 5 | Receipt-backed compaction | feat6 | `compaction.py`, `context.py` |
| 6 | Steering + public status + exactly-once reconcile | feat7+8 | `control.py`, `status.py`, `reconcile.py` |
| 7 | Equivalence harness | feat9 | `harness.py`, `evaluation.py` |

Extension (not a day): COW/radix cache — `docs/week4-cow-radix-extension-plan.md`.

## Design note: simplified workspace (Day 2)

The old 7-day workspace carried a write-ahead mutation journal and undo
machinery (old Day 6 content). The new design drops that machinery: the
workspace keeps the authorization core (bounds, protected paths, observed
digests, approvals, atomic writes) plus effect receipts; crash/effect
recovery is taught by Day 6's exactly-once reconcile instead. This keeps each
day's surface small and matches the "start simple, extend" arc.

## Stack shape

- PR 1: remove the old 7-day refsol; create the starter skeleton mapped to the
  new refsol (this is the map reviewers read).
- PRs 2-8: implement each day's feature(s) in the refsol + focused tests.
- PR 9: COW/radix plan (design only).

## Why 7 days

Matches the existing published `week4-01..07` structure so book numbering stays
stable; balances learner load (checkpoint+rewind share Day 4 as one
"checkpoint, branch, undo" story; steering/status/reconcile share Day 6 as one
"control, inspect, recover" story). COW is a scaling mechanism, not a
correctness lesson, and is deferred to the extension.
