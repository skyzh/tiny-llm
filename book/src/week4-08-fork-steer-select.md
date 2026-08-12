# Day 8: Fork, Steer, and Select

> 🚧 **Early-review WIP:** Use only pre-created disposable workspaces. A
> control-state fork does not undo a file edit or any other completed effect.

Days 4 and 6 paused one agent and resumed one continuation. Day 8 asks a new
question: after the model has inspected or changed the workspace, can we reuse
the same inference prefix, try two explicit directions, and select the branch
whose observable result is better?

The answer reconnects Week 4 to the inference system from Weeks 1–3. The course
tokenizer renders the checkpoint conversation once. `TinyKvFullCache` stores
the prefix keys and values for every model layer. Each branch gets a fresh
control object and cache handles that share only those immutable prefix arrays,
then decodes its own suffix. The branch report exposes the reused token count,
the layer offsets, and the full-prefix prefill that was avoided.

This is control-state reuse, not effect rollback. Copy the already-modified
disposable workspace and its completed receipt log before running either
branch. Both copies begin with the same files and evidence; later receipts stay
inside their branch.

## The Starter Surface

Day 8 adds one module and extends the approval result:

| File | Public names | Purpose |
| --- | --- | --- |
| `src/tiny_llm/agent/workspace.py` | `ApprovalDecision` | Carry an operator's denial reason back as one ordinary model-visible observation. |
| `src/tiny_llm/agent/branching.py` | `PrefixReuse`, `KvPrefixGenerator`, `BranchOutcome`, `run_branch`, `select_branch` | Reuse a dense KV prefix, run isolated steered continuations, evaluate them, and make one explicit choice. |
| `src/tiny_llm/agent/__init__.py` | the names above | Export the cumulative Day 8 API. |

Copy and run the five learner tasks:

```bash
pdm run copy-test --week 4 --day 8
pdm run test --week 4 --day 8
```

Use this command for the supplied implementation:

```bash
pdm run test-refsol --week 4 --day 8
```

Before you implement the TODOs, all five Day 8 tasks are expected to fail.

## Task 1: Return a Reason with a Denial

Add the immutable decision:

```python
ApprovalDecision(approved=False, reason="keep the requested answer at 2")
```

A structured denial requires a nonblank reason. `Workspace.execute` returns
that reason in its normal `error:` result so the next model turn can react to
the operator's instruction. It does not execute the effect or append a receipt.
Existing callbacks that return plain `True` or `False` remain compatible.

The reason is steering, not a secret channel. Keep it short and suitable for
the model-visible transcript.

## Task 2: Save One Real Token and KV Prefix

`KvPrefixGenerator.save_checkpoint(messages)` renders the checkpoint messages
without a generation prompt, tokenizes them with the course tokenizer, and
prefills one `TinyKvFullCache` per layer. It records the exact token IDs and
layer offsets in the existing Day 4 `ModelCheckpoint`.

The saved prompt must be an exact token prefix of every later steered prompt.
Reject a continuation if even a same-length token differs. This binds cache
reuse to content, not merely to a position.

`fork()` creates a fresh generator whose cache handles point at the frozen
prefix arrays. When one branch grows, `TinyKvFullCache` assigns newly
concatenated arrays to that branch. The frozen prefix and its sibling remain
unchanged. This lesson deliberately uses the dense compatibility path; paged
copy-on-write and radix serving are separate scaling topics.

## Task 3: Expose What Was Reused

Each continuation reports:

```python
PrefixReuse(
    reused_tokens=prefix_length,
    layer_offsets=(prefix_length, ...),
    avoided_prefill_tokens=prefix_length,
)
```

The first suffix model call starts at `prefix_length`; it must not call the
model again at offset zero. These numbers make the inference boundary visible:
the branch is not cloning only a Python transcript and silently recomputing the
whole prompt.

## Task 4: Fork Effects and Evidence Explicitly

Suppose the completed prefix changed `app.py` from `answer = 1` to
`answer = 2` and wrote `call-1`, the edit receipt. Copy both the post-effect
workspace and `receipts.jsonl` into two roots:

```text
base after checkpoint
├── app.py              answer = 2
└── receipts.jsonl      call-1: edit_file

validate-only/          try-extra-edit/
├── app.py              ├── app.py
└── receipts.jsonl      └── receipts.jsonl
```

Both receipt files begin byte-identical and contain `call-1`. The
`validate-only` branch appends `call-2` after its exact allowed validation
command. The other branch asks to change the answer again; the operator denies
it with a reason, so its file and receipt bytes remain unchanged.

Construct each branch with its own `Workspace` and `ReceiptStore`, then call:

```python
outcome = run_branch(
    "validate-only",
    "validate without another edit",
    checkpoint,
    prefix_generator.fork(),
    workspace,
    receipts,
    evaluation_case,
)
```

`run_branch` composes the Day 6 steered resume with the Day 7 observable-outcome
evaluator. It does not copy a directory, infer an evaluation case, or merge
effects for you.

## Task 5: Select One Passing Branch

Make the choice explicit:

```python
selected = select_branch(outcomes, "validate-only")
```

The name must identify exactly one outcome, and that outcome must pass its Day
7 report. Reject an absent name, duplicate names, or a failing branch. Day 8
does not invent a hidden score or ask another model to judge the traces.

## Manual Qwen/MLX Walkthrough

Complete Weeks 1–3 and the Day 8 TODOs first. Use a cached local Qwen model and
the same dense compatibility path:

```python
from mlx_lm import load
from tiny_llm import Qwen3ModelWeek3
from tiny_llm.agent import KvPrefixGenerator

mlx_model, tokenizer = load("Qwen/Qwen3-0.6B-MLX-4bit")
model = Qwen3ModelWeek3(mlx_model, enable_paged_attention=False)
generate = KvPrefixGenerator(model, tokenizer, max_tokens=128)
```

Create the Day 4 checkpoint after a complete tool observation, call
`generate.save_checkpoint(messages)` for that exact message list, and fork two
fresh generators. Give them different visible steering messages and the two
workspace/receipt copies described above. Print each `outcome.reuse`, render
both evaluation reports, and select the passing name.

Model responses are nondeterministic, so this walkthrough is manual. Inspect
the actual proposed actions, approval reason, final file bytes, receipt logs,
and evaluation reports. The deterministic learner test covers the same public
boundary with a tiny tokenizer/model and no download.

## Checkpoint

You can now connect a Day 4 control checkpoint to the actual tokenizer and KV
cache path, reuse one immutable prefix for two isolated continuations, expose a
denial reason to the model without recording an effect, evaluate both branches
from declared evidence, and choose one passing result.

Completed effects were copied, not rewound. The two branches do not run
concurrently, share a mutable workspace, merge receipts, or provide a session
server/tree. Day 8 teaches the boundary visibly before adding any serving-scale
machinery.

{{#include copyright.md}}
