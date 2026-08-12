# Week 4 Day Split (reference for reviewers)

Status: Days 1--8 are published checkpoints. Each day ships as one cumulative
learner PR so reviewers can see exactly what belongs to that checkpoint.

## 8-day structure

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

Extension (not a day): COW/radix cache — `docs/week4-cow-radix-extension-plan.md`.

## Delivery shape

- Each learner day is one complete Draft PR based on the previously merged day.
- The PR includes the reference, solution-free starter, focused course-code
  tests, and learner chapter for that checkpoint.
- Days are implemented sequentially. A later day does not leak API or prose
  into the current starter.

## Why 8 days

The first seven days establish the agent loop and its observable evidence. Day
8 reconnects that control path to the tokenizer and KV cache built in Weeks
1--3. Each day adds one visible concept; scaling and production-hardening
machinery stay outside the core course unless a later checkpoint explicitly
teaches it.
