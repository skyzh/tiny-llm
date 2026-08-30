# Day 5: Compact Completed Work

> 🚧 **Early-review WIP:** This chapter is public for early review and may
> change. Use a disposable workspace when running the agent or enabling writes
> or commands.

Every tool call adds two model-visible messages: the assistant's action and the
tool observation. A long validation log can eventually crowd out the task and
the useful recent steps. Deleting old messages saves space, but it also deletes
the evidence behind claims such as “validation passed.”

Day 5 makes one boundary visible: replace an older, completed effect with a
small deterministic evidence record while keeping its full `EffectReceipt`
unchanged. The model receives fewer tokens and can continue from that view; the
harness still retains the exact action and result.

## The Starter Surface

The cumulative Day 9 scaffold already contains later steering, evaluation,
branching, and evidence declarations. Leave those future TODO bodies alone.
Day 5 owns one small module:

| File | Public names | Purpose |
| --- | --- | --- |
| `src/tiny_llm/agent/compaction.py` | `CompactionResult`, `compact_completed_interactions` | Derive a smaller model-visible transcript from completed, receipted effects. |
| `src/tiny_llm/agent/__init__.py` | the names above | Complete the Day 5 exports within the final scaffold. |

Copy the learner test, then run it:

```bash
pdm run copy-test --week 4 --day 5
pdm run test --week 4 --day 5
```

Use this command for the supplied implementation:

```bash
pdm run test-refsol --week 4 --day 5
```

Before you implement the TODOs, all six Day 5 tasks are expected to fail.
The command runs only the Day 5 test. `copy-test` refreshes the learner copy
from the supplied checkpoint.

## Start From the Existing Transcript

The input is the same list of role/content messages that `run_agent` gives the
model. A completed effect has this shape:

```python
[
    {"role": "assistant", "content": '{"tool":"run_command",...}'},
    {"role": "user", "content": "Tool result:\nstatus: 0\n..."},
]
```

The compactor does not invent a second event log. It receives this transcript
plus the Day 3 `EffectReceipt` values already produced by the workspace.

Before implementing it, predict which older interaction in the focused
fixture is eligible to compact, which recent interaction must stay verbatim,
and whether `saved_tokens` must be positive. The returned messages, exact
counter values, and unchanged receipts let you falsify that prediction.

## Task 1: Require Exact Receipt Evidence

A pair is eligible only when all three facts match one supplied receipt:

1. the parsed tool name;
2. the normalized argument object; and
3. the complete observation text.

If there is no receipt, or if any of those fields differs, leave both messages
verbatim. This deliberately excludes Day 2 reads and listings: the current
course receipts effects, not every observation. Day 5 must not pretend that an
unreceipted result is durable evidence.

## Task 2: Keep a Small, Honest Record

Keep the small assistant action, but replace its large tool-observation message
with one bounded evidence record that contains:

- the tool and its normalized arguments;
- `exit_state` and `changed_artifacts`;
- a bounded prefix of the result; and
- the content-addressed `receipt_id`.

For example:

```text
Completed tool interaction (compacted evidence):
{"arguments":{"argv":["python","validate.py"]},
 "changed_artifacts":[],"exit_state":"ok",
 "receipt_id":"...","result_preview":"status: 0...",
 "tool":"run_command"}
```

This is not a model-written summary. It is a deterministic rendering of fields
the harness already verified. The bounded preview helps the model explain what
happened; `receipt.result` still contains the full observation.

## Task 3: Retain a Recent Tail

`keep_recent=1` leaves the newest eligible effect as its original two messages.
Older matching effects may compact. The recent tail keeps the next decision
grounded in the exact latest interaction without requiring a complicated
semantic policy.

The count refers only to receipted effect interactions. Unreceipted reads stay
verbatim regardless of this setting.

## Task 4: Measure the Real Model Input

The compactor accepts `count_tokens(messages)` instead of estimating tokens
from characters. For the real model, use the same tokenizer and chat template
as generation:

```python
def count_tokens(messages):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(tokenizer.encode(prompt, add_special_tokens=False))
```

Build the proposed compact view, count it again, and accept it only when the
exact counter decreases. `CompactionResult` reports `tokens_before`,
`tokens_after`, and `saved_tokens`. The focused tests inject the Day 4 fake
model's simple counter, so they do not load or download a model.

## Task 5: Preserve the Source of Truth

Never mutate the caller's message list or any receipt. Return copied messages
inside `CompactionResult`. Running the compactor again over its own output is a
no-op because a compact evidence record is not a tool action followed by a tool
result.

This distinction matters:

```text
original transcript + full receipts   durable evidence owned by the harness
                  |
                  v
       compacted message view          temporary input for the model
```

If the compact view is lost, derive it again. Do not treat it as a replacement
for receipts or checkpoints.

## Task 6: Continue With the Existing Model Boundary

`CompactionResult.messages` contains ordinary role/content mappings. Pass a
copied list to the same generation callable used by the loop, then validate the
response through the existing protocol:

```python
view = compact_completed_interactions(
    messages,
    receipts,
    count_tokens,
    keep_recent=1,
)
response = generate([dict(message) for message in view.messages])
action = parse_action(response, workspace.available_tools)
```

The Day 5 test makes the scripted model return a final answer after it sees the
compact validation evidence. Nothing about action parsing, tool approval, or
workspace execution changes.

## Limits of This Teaching Compactor

This checkpoint intentionally does not add semantic-perfect summarization,
automatic threshold scheduling inside `run_agent`, persistent compact views,
receipt lookup by summary text, K/V cache editing, session trees, rewind,
steering, or exactly-once execution. It compacts only completed effects backed
by the receipts the caller supplies.

## Checkpoint

You can now make an older validation interaction visibly smaller, inspect the
receipt that retains its complete result, and feed the compact view to the next
model call. Continue with [Day 6: Inspect and Steer a Paused
Agent](week4-06-steering.md) to inspect one checkpoint, add one visible operator
message, and resume without replaying completed work.

The current Day 6 path resumes the Day 4 transcript; it does not yet consume
`CompactionResult.messages`. The pending cumulative capstone will carry this
compacted view into the later control path. Until then, Day 5's token counts
prove only deterministic transcript accounting, not latency, throughput,
quality, or memory-capacity improvement.

{{#include copyright.md}}
