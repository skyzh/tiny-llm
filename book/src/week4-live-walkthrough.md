# 🚧 Week 4 Live Walkthrough: One Real Model Through Five Modules

<!-- DRAFT for TINYLLM-928. Every line marked TBD-927 waits for Forge's
command and output contract from TINYLLM-927; nothing here is verified yet. -->

The nine day checks prove your mechanisms with scripted models. This
walkthrough shows the same mechanisms working with a real model: the
supplied Week 4 implementation on the Week 2 reference engine and a cached
Qwen3-4B, through all five modules of the week. It doesn't run your code, so
it works at any point. It makes the most sense after Day 9, once you have
built every piece it uses.

| Module | Days | What you watch the real model do |
| --- | --- | --- |
| Control | 1 | Propose one JSON action at a time, through `generate_response()` and a fresh growing cache |
| Effects | 2, 3 | Read the fixture, make the approved edits, and leave receipts |
| Sessions | 4, 6 | Pause after a complete observation, take a steering message, and resume |
| Context | 5, 9 | Compact completed work and read oversized evidence by range, measured in real tokens |
| Branches | 7, 8 | Continue twice from one shared prefix, then accept the branch you name only if its Day 7 report passes |

The [deterministic capstone](week4-overview.md#work-through-one-chapter)
(`pdm run week4-capstone`) and the
[confined fixture](week4-02-tools.md#inspect-the-supplied-fixture)
(`--autoreject-fixture`) stay as they are. They are the repeatable checks.
This walkthrough is the live run, and model choices vary from run to run.

## Before You Start

- Complete [Setup](setup.md) and build the reference extension with
  `pdm run build-ext-ref`.
- Have the `Qwen/Qwen3-4B-MLX-4bit` snapshot in your local Hugging Face
  cache from Setup. The walkthrough never downloads. If the snapshot is
  missing, it stops with an error instead of fetching it.

Run it from the checkout root:

```bash
PDM_CHECK_UPDATE=false pdm run --skip pre_run agent -- --live-fixture
```

It starts in a fresh temporary workspace that holds only the fixture
files. It allows only the named reads and edits, validates fixture bytes
with a fixed in-process check, and refuses anything else before it has an
effect. Tool calls and generated tokens both have fixed budgets.

## Control and Effects

The model proposes actions, and your harness decides what happens. Look
for:

- TBD-927: the action and observation lines for the first read, and the
  exact `app.py` bytes it returns.
- TBD-927: the approved edits, the refused unapproved effect, and the
  validation result.
- TBD-927: where the receipts are written and what each receipt records.

If the model answers before it has read the file, the run stops with an
error. A guess never counts as success.

## Sessions: Pause, Steer and Resume

The run pauses after a complete action and observation, never in the
middle of a tool call. At the `pause>` prompt, inspect the completed
observation, then type `resume` to continue from the same conversation.
Steering comes later. Each branch adds its own visible, fixed Day 6
steering message to the checkpoint before it continues.

- **How long the checkpoint lasts:** it lives inside the running process
  until you resume it or the process exits. Exit at `pause>` and the
  checkpoint is gone. There is no resuming it from a new process.
- **What to check after resume:** the resumed conversation is the saved
  one plus your steering message, and no completed effect runs again. The
  receipts from before the pause are unchanged. TBD-927: the lines that
  show this.

## Context: Real Token Counts

The capstone counts characters. This run counts the model's own tokens.
Look for the `context>` lines.

- **Compaction:** TBD-927: the tokens before and after, measured with the
  Week 2 tokenizer, and the receipts that stay exactly as they were.
- **Oversized evidence:** TBD-927: the artifact size, the bytes the model
  sees, and the exact range it reads back.

## Branches: Shared Prefix, Separate Effects, Your Choice

The run continues twice from one checkpoint, each branch with its own
steering message. The `validate-only` branch
validates without another edit. The `add-notes` branch writes a fixed note
first, then validates. Look for one `branch>` block per branch, with that
branch's effects, receipts, observed cache offsets and Day 7 report.

- **Cache reuse:** both branches start from the same KV prefix, built by
  the Week 2 tokenizer and cache. TBD-927: the reused-token count and layer
  offsets the run actually observes.
- **Separate effects:** each branch works in its own copy of the
  workspace, with its own receipt log. TBD-927: where each branch's files
  and receipts appear.
- **Selection:** you name the branch you want. The run accepts it only if
  that branch's Day 7 report passes every check. A named branch that fails
  its report is refused, as on [Day 8](week4-08-fork-steer-select.md). The
  run never picks a winner for you. At the `select>` prompt, after
  you have seen both reports, type `validate-only` or `add-notes`. TBD-927: the accepted and refused
  output.

## What This Run Proves, and What It Doesn't

It shows the week's mechanisms working around a real model: effects confined to
the fixture, receipts that survive a pause, real token and cache
accounting, and a selection the Day 7 report gates. It doesn't prove
the model is good at the task, and it doesn't measure speed. Different
runs can make different choices, so use the day checks and the capstone
for repeatable results.

{{#include copyright.md}}
