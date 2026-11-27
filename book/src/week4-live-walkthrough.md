# 🚧 Week 4 Live Walkthrough: One Real Model Through Five Modules

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

## What You Will See

The run prints one labelled line for each stage, such as `pause>`,
`context>`, `branch>` and `selected>`. Most carry a JSON object. The
numbers below come from one verified run. A real model can phrase its
actions differently from run to run, so your counts may differ slightly.
What stays the same is which checks pass and which effects are allowed.

## Control and Effects

The model proposes actions, and the harness decides what happens. In the
fixture, the model reads `app.py`, makes the approved edit and validates
it. A write it wasn't approved to make is refused before anything is
written. The run then pauses and prints a `pause>` line that summarizes
this first stage:

- `last_action`: the model's last action, a `read_file` of `app.py`.
- `last_observation`: the `app.py` text that read returned, which
  includes `Welcome`.
- `denied_unapproved_write`: `true`, for the refused note.
- `receipts`: the edit and validation receipts. Each records an approved
  effect.

If the model answers before it has read the file, the run stops with an
error. A guess never counts as success.

## Sessions: Pause and Resume

The run pauses after a complete action and observation, never in the
middle of a tool call. After the `pause>` line it prints
`resume> type resume to continue`. Inspect the paused state, then type
`resume`.

- **How long the checkpoint lasts:** it lives inside the running process
  until you resume it or the process exits. Exit at the prompt and the
  checkpoint is gone. There is no resuming it from a new process. Typing
  anything other than `resume` stops the run.
- **What resume keeps:** the run continues from the saved conversation,
  and no completed effect runs again. Before you choose a branch, the run
  checks that the original workspace and receipt log are unchanged.

Steering comes in the next stage. Each branch adds its own visible, fixed
Day 6 steering message to the checkpoint before it continues.

## Context: Real Token Counts

The capstone counts characters. This run counts the model's own tokens.

- **Compaction:** the first `context>` line shows `tokens_before` and
  `tokens_after`, measured with the Week 2 tokenizer, plus the
  `receipt_ids` the compacted view keeps. In the verified run it went from
  671 to 494 tokens.
- **Oversized evidence:** each branch prints a `context>` line with its
  `branch` name. It shows the full `artifact_bytes`, the `inline_bytes` the
  model sees, and the exact `range_bytes` and `range_text` it reads back. In
  the verified run, both branches stored 1,053 bytes, showed the model 512,
  and read back the exact first 63 bytes.

## Branches: Shared Prefix, Separate Effects, Your Choice

The run continues twice from the compacted checkpoint, each branch with its
own steering message. The `validate-only` branch validates without another
edit. The `add-notes` branch writes a fixed note first, then validates.
Each prints a `branch>` line.

- **Cache reuse:** `reused_tokens` and the `cache` object's
  `layer_offsets` show that both branches started from the same KV prefix,
  built by the Week 2 tokenizer and cache. In the verified run, each branch
  reused the whole 494-token compacted prefix at all 36 layers.
- **Separate effects:** each branch works in its own copy of the workspace,
  with its own receipt log. `changed` lists the files that branch modified,
  and `receipts` its effects. In the verified run, `validate-only` wrote no
  new file and added one validation receipt. `add-notes` wrote `NOTES.md` and
  added write and validation receipts. Before you choose, the run checks
  that neither branch touched the original workspace or the other branch.
- **Report:** `report` is that branch's Day 7 evaluation, and `passed` is
  its result. In the verified run, `validate-only` passed 8 checks and
  `add-notes` passed 10.
- **Selection:** after both reports, the run prints
  `select> type validate-only or add-notes`. Type the branch you want. The
  run accepts it only if that branch's Day 7 report passes, as on
  [Day 8](week4-08-fork-steer-select.md). A failing branch is refused with
  an error. The run never picks a winner for you. An accepted choice
  prints, for example, `selected> {"name": "validate-only", "passed": true}`.
  In the verified run both branches passed, so either name would have been
  accepted.

## What This Run Proves, and What It Doesn't

It shows the week's mechanisms working around a real model: effects confined to
the fixture, receipts that survive a pause, real token and cache
accounting, and a selection the Day 7 report gates. It doesn't prove
the model is good at the task, and it doesn't measure speed. Different
runs can make different choices, so use the day checks and the capstone
for repeatable results.

{{#include copyright.md}}
