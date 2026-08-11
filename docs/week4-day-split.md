# Week 4 Day Split (reference for reviewers)

Status: decided by Forge per Chi's instruction ("7/10/x days your call"). The
refsol stack is feature-based (one PR per feature so reviewers see exactly
what belongs to each review boundary); the 7-day split groups adjacent
features into teaching days. Curriculum prose (book) is not part of this
stack.

## 7-day structure

| Day | Theme | Features (PRs) | Modules |
|---|---|---|---|
| 1 | Validated agent loop + tool protocol | feat1 | `protocol.py`, `loop.py`, `generation.py` (minimal) |
| 2 | Authorize effects + durable receipts | feat2 | `workspace.py` (simplified), `receipts.py` |
| 3 | Session tree (event-level id/parentId) | feat3 | `session.py` |
| 4 | Derived KV state: checkpoint/resume + sequential rewind | feat4 + feat5 (two PRs) | `checkpoint.py`, `branch.py`, `generation.py` |
| 5 | Receipt-backed compaction | feat6 | `compaction.py` |
| 6 | Control/inspect/recover: steering + status, then exactly-once reconcile | feat7 + feat8 (two PRs) | `control.py`, `status.py`, `reconcile.py` |
| 7 | Equivalence harness | feat9 | `harness.py` |

Extension (not a day): COW/radix cache — `docs/week4-cow-radix-extension-plan.md`.

The old static-held-out grader (`evaluation.py` with `TaskPackage`/`StagedTask`)
is not part of the new course: Day 7's `harness.py` measures the integrated
system (warm/cold, fork/cold, compact/full, crash/resume equivalence) over
the three planes, which replaces the old standalone grader as the evaluation
story.

## Design note: simplified workspace (Day 2)

The old 7-day workspace carried a write-ahead mutation journal and undo
machinery (old Day 6 content). The new design drops that machinery: the
workspace keeps the authorization core (bounds, protected paths, observed
digests, approvals, atomic writes) plus effect receipts; crash/effect
recovery is taught by Day 6's exactly-once reconcile instead. This keeps each
day's surface small and matches the "start simple, extend" arc.

The old summarizer-based `ContextManager` (whole-history compaction with a
model summary) is not part of the new course: Day 5's receipt-backed
compaction replaces it, keeping the durable trace untouched and re-expanding
verified ranges on demand. This removes a large control-coupled module and
keeps Day 5 self-contained.

## Stack shape (11 PRs)

1. reset: remove the old 7-day refsol; create the starter skeleton mapped to
   the new refsol (this is the map reviewers read).
2. loop + tool protocol (Day 1)
3. effect receipts (Day 2)
4. session tree (Day 3)
5. KV checkpoint (Day 4a)
6. sequential rewind (Day 4b)
7. receipt-backed compaction (Day 5)
8. steering/status (Day 6a)
9. exactly-once reconcile (Day 6b)
10. equivalence harness (Day 7)
11. COW/radix plan (design only, non-day extension)

Each feature PR is independently reviewable; the day mapping above shows how
adjacent features group into teaching days.

## Why 7 days

Matches the existing published `week4-01..07` structure so book numbering stays
stable; balances learner load (checkpoint+rewind share Day 4 as one
"checkpoint, branch, undo" story; steering/status/reconcile share Day 6 as one
"control, inspect, recover" story). COW is a scaling mechanism, not a
correctness lesson, and is deferred to the extension.
