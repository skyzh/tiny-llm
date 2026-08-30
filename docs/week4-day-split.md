# Week 4 Day Split (reference for reviewers)

Status: Days 1--9 are published checkpoints. The earlier per-day cumulative-PR
model is retired. The repository now ships one cumulative Day 9 declaration
scaffold; this document records which visible prefix each day owns.

## 9-day structure

| Day | Theme | Features (PRs) | Modules |
|---|---|---|---|
| 1 | Validated agent loop + tool protocol | feat1 | `protocol.py`, `loop.py` |
| 2 | Inspect a workspace | read-only list/read tools | `workspace.py` |
| 3 | Edit, validate, and record | approved edits, one command, simple receipts | `workspace.py`, `receipts.py` |
| 4 | Checkpoint and resume | one conversation + fake-model cache snapshot | `checkpoint.py`, `loop.py` |
| 5 | Compact completed work | bounded receipt-backed transcript view | `compaction.py` |
| 6 | Inspect and steer | safe-boundary status and one visible steering message | `steering.py` |
| 7 | Evaluate outcomes | declared final/file/result/receipt facts | `evaluation.py` |
| 8 | Fork, steer, and select | dense tokenizer/KV prefix reuse, isolated branches, explicit selection | `branching.py`, `workspace.py` |
| 9 | Bound tool evidence | exact external bytes, bounded observation, explicit range retrieval | `evidence.py` |

Extension (not a day): COW/radix cache — `docs/week4-cow-radix-extension-plan.md`.

## Delivery shape

- The solution-free starter exposes the public declarations for all nine days.
  A learner implements only the prefix assigned by the current chapter and
  leaves later-day TODO bodies alone.
- Each day has a focused, day-local course-code test. It does not rerun earlier
  days or prove the cumulative product path.
- The supplied reference implements all nine days. Starter/reference API-sync
  checks keep the visible declarations aligned while the chapters preserve
  day-by-day ownership.

## Why 9 days

The first seven days establish the agent loop and its observable evidence. Day
8 reconnects that control path to the tokenizer and KV cache built in Weeks
1--3. Day 9 keeps large observable evidence available without filling every
later model prompt. Each day adds one visible concept; scaling and
production-hardening machinery stay outside the core course unless a later
checkpoint explicitly teaches it.
