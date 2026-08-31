# Day 6: Inspect and Steer a Paused Agent

> 🚧 **Early-review WIP:** This chapter is public for early review and may
> change. Use a disposable workspace when running the agent or enabling writes
> or commands.

Day 4 gave the harness a safe pause after a complete tool observation. Day 5
showed how to make older completed evidence smaller for the model without
discarding the full receipts. Those boundaries also create a useful moment for
an operator: inspect what the paused run has actually recorded, add one new
instruction, and let a fresh model continue.

The focused Day 6 scenario resumes the original Day 4 transcript rather than
`CompactionResult.messages`. After Day 9, the supplied deterministic capstone
connects the compacted view to this later control path.

Day 6 implements only that interaction. It does not inspect hidden reasoning or
guess the model's plan. It reports public facts from the checkpoint, appends one
ordinary steering message, and resumes through the existing validated loop.

## The Starter Surface

Later evaluation, branching, and bounded-evidence declarations are already
visible in the final Day 9 scaffold. Leave those TODO bodies alone. Day 6 owns
one module:

| File | Public names | Purpose |
| --- | --- | --- |
| `src/tiny_llm/agent/steering.py` | `AgentStatus`, `inspect_checkpoint`, `resume_with_steering` | Inspect one complete-observation checkpoint, append one operator message, and resume. |
| `src/tiny_llm/agent/__init__.py` | the names above | Complete the Day 6 exports within the final scaffold. |

Run the cumulative learner checkpoint:

```bash
pdm run test --week 4 --day 6
```

Use this command for the supplied implementation:

```bash
pdm run test-refsol --week 4 --day 6
```

Before you implement the TODOs, all six Day 6 tasks are expected to fail. The
command force-refreshes and runs the supplied learner tests for Days 1--6
together.

## Start at a Safe Pause

`run_to_checkpoint(...)` saves after the assistant action and its tool result
are both present:

```text
original task
    ...
assistant tool action
tool observation
                  ^ inspect and steer here
```

This is deliberately not mid-token or mid-tool steering. No process is running
in the background. The workspace is quiescent, and the checkpoint binds the
original task, the complete message prefix, and the fake-model cache metadata.

## Task 1: Derive a Public Status

Implement:

```python
inspect_checkpoint(checkpoint, evidence_chars=160) -> AgentStatus
```

Validate the checkpoint, then require its final two messages to be a parsed
assistant tool action followed by a `Tool result:\n...` user observation.
Return four public facts:

```python
AgentStatus(
    task="fix app.py and validate",
    last_action='{"new":"2","old":"1","path":"app.py","tool":"edit_file"}',
    last_evidence="edited app.py",
    next_step="resume the model after the completed edit_file observation",
)
```

`last_evidence` is a bounded prefix of the recorded observation. When it is too
long, reserve the final character for an ellipsis so its length never exceeds
`evidence_chars`.

The status function must not receive or call a model or workspace. It reads the
validated immutable checkpoint and returns a new frozen value.

## Task 2: State Only What the Harness Knows

The checkpoint knows the current task and the last completed action/result. It
does not know what the model will decide next. Therefore `next_step` is a
deterministic harness boundary:

```text
resume the model after the completed edit_file observation
```

Do not replace this with a semantic guess such as “update the tests next.” That
would present invented intent as agent state. A richer plan would need its own
explicit, model-visible artifact; Day 6 does not add one.

Reject an incomplete boundary rather than producing a misleading card. A
checkpoint ending with an ordinary user message, malformed assistant text, or
anything other than a complete action/observation pair is not inspectable by
this API.

## Task 3: Append One Steering Message

Implement:

```python
resume_with_steering(
    checkpoint,
    steering,
    fresh_generate,
    workspace,
    limits=None,
) -> AgentRun
```

Reject an empty or whitespace-only instruction. For a valid instruction,
restore the checkpoint into the fresh generator and derive a mutable copy of
the saved transcript. Append exactly one ordinary user message:

```python
{
    "role": "user",
    "content": "Operator steering:\nvalidate before answering",
}
```

Then pass that list to the existing bounded loop. Do not add a separate queue,
control channel, hidden prompt, or special protocol action.

## Task 4: Keep the Message in Stable Order

The steering message belongs immediately after the saved checkpoint prefix.
If the resumed model chooses another tool, the existing loop appends that
assistant action and its observation after the steering:

```text
saved task and evidence
Operator steering: validate before answering
assistant run_command action
validation observation
assistant final answer
```

Append the steering message once. Because the loop carries its message list
forward, every later model call sees the same single message at the same
position. Reinserting it on every call would duplicate the instruction and
change the conversation.

## Task 5: Continue Without Replaying Effects

The focused scenario starts with the original task “fix app.py and validate.”
The first model reads and edits `app.py`, and the harness checkpoints after the
complete edit observation. At that point:

- `app.py` already contains the new value;
- the edit approval happened once;
- the receipt store contains the edit receipt; and
- inspection reports the saved edit evidence without touching the workspace.

The operator adds “validate before answering.” A fresh scripted model restores
the saved prefix, sees the original task, edit evidence, and steering message,
then runs the exact allowed validation command and returns its final answer.
The test proves the edit approval remains single, validation executes once, and
the edit and command retain separate receipts.

Steering changes the next model input. It does not undo, replay, or rewrite the
completed prefix.

## Task 6: Keep the Boundary Small

Fail clearly when the checkpoint identity is invalid, the generator cannot
restore the saved fake-model state, the evidence limit is not a positive
integer, or steering is blank. Reuse `AgentError`, `AgentCheckpoint`,
`AgentLimits`, `AgentRun`, and the existing loop rather than creating parallel
versions.

The Day 6 module does not add concurrent interruption, mid-token control, a
background worker or status server, a durable steering queue, session trees,
branch/rewind, exactly-once reconciliation, or an evaluator. It is one visible
pause → inspect → steer → resume path for the course's scripted model.

## Checkpoint

You can now pause after a complete observation, show an operator a bounded
status made only from recorded facts, add one visible instruction, and resume a
fresh model without replaying the completed effect. Inspect the model inputs in
the focused test to verify the original task, saved evidence, and steering stay
in order through a later tool turn and final answer.

Continue with [Day 7: Evaluate Observable Outcomes](week4-07-evaluation.md) to
turn the final workspace, tool results, and durable receipts into a structured
pass/fail report without grading hidden reasoning or exact transcript shape.

That report is another deterministic library checkpoint. After Day 9, the
supplied Week 4 capstone provides the runnable path that carries checkpoint,
compaction, steering, evaluation, branch selection, and bounded evidence
together.

{{#include copyright.md}}
