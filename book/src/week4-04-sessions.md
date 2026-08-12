# Day 4: Checkpoint and Resume

Days 1 through 3 build one uninterrupted coding-agent run. The model proposes
an action, the harness executes it, and the result becomes the next model
observation. But if the process stops, a new model object does not know which
conversation prefix it had already processed.

Day 4 makes one boundary visible: save the conversation and the small model
snapshot immediately after a complete tool observation, then restore both into
a fresh scripted model and continue at the next response. The completed tools
stay completed; resume starts after their observations instead of replaying
them.

This is a teaching checkpoint for one process and the course's fake model. It
is not a session tree, rewind feature, persistent KV store, transaction log, or
exactly-once effect system.

## Check the Chapter

Implement the TODO-only surfaces in:

| File | Public names | Responsibility |
| --- | --- | --- |
| `src/tiny_llm/agent/checkpoint.py` | `ModelCheckpoint`, `AgentCheckpoint`, `create_checkpoint` | Represent and validate one in-memory conversation/model snapshot. |
| `src/tiny_llm/agent/loop.py` | `run_to_checkpoint`, `resume_agent` | Stop after a complete observation, then continue with a fresh model. |
| `src/tiny_llm/agent/__init__.py` | the names above | Export the cumulative Day 4 API. |

Run the learner checkpoint from the repository root:

```bash
pdm run test --week 4 --day 4
```

Before you implement the TODOs, all seven Day 4 tasks are expected to fail.
The test uses a scripted model, fake cache metadata, a temporary workspace, and
one exact Python validation command. It does not load model weights.

Course maintainers can check the supplied implementation without copying the
learner test:

```bash
pdm run test-refsol --week 4 --day 4
```

## One Safe Loop Boundary

A checkpoint is saved only after the harness has appended both halves of a
tool interaction:

```text
assistant: {"tool":"edit_file", ...}
user:      Tool result:\nedited app.py
                                      ^ checkpoint here
```

Saving before the observation would leave the restored model unable to tell
whether the tool ran. Day 4 therefore counts completed tool calls and saves at
the boundary after `_append_tool_result(...)` has produced the next complete
conversation.

The checkpoint stores the semantic messages, not the `AgentEvent` history.
Day 3 receipts remain separate evidence about the edit or command. They are not
copied into the checkpoint and they do not become a replay controller.

## Task 1: Represent the Fake Model Snapshot

`ModelCheckpoint` contains four fields, in order:

```python
conversation_position: int
response_index: int
cached_token_ids: tuple[int, ...]
layer_offsets: tuple[int, ...]
```

`conversation_position` is the number of semantic messages at the saved
boundary. `response_index` tells the scripted model which response comes next.
`cached_token_ids` represents the prompt prefix in the fake cache, and every
`layer_offsets` entry must equal its length.

Reject negative positions, invalid token IDs, a missing layer snapshot, or
offsets that disagree with the cached prefix. These checks make the fake model
state internally coherent without introducing a production cache format.

## Task 2: Bind Conversation and Model State

`AgentCheckpoint` contains:

```python
checkpoint_id: str
task: str
messages: tuple[tuple[str, str], ...]
model: ModelCheckpoint
```

`create_checkpoint(task, messages, model)` copies each mutable message into an
immutable `(role, content)` pair. It computes `checkpoint_id` as the SHA-256 of
canonical JSON containing the task, messages, and model fields.

`AgentCheckpoint.validate()` checks the ordinary resume contract:

- the task and messages are structurally valid;
- the model's conversation position equals the saved message count;
- the checkpoint ID still matches the content.

This identity check catches an accidental mismatch. It is not a hostile-tamper
or authentication scheme.

## Task 3: Stop After a Complete Observation

`run_to_checkpoint(task, generate, workspace, after_tool_calls=1, limits=None)`
starts with the same prompt and validation rules as `run_agent`. It counts a
tool call only after execution and observation append. At the requested count,
it calls:

```python
model_state = generate.save_checkpoint(messages)
```

and returns an `AgentCheckpoint`.

The generator must return a `ModelCheckpoint`. A missing checkpoint method, a
non-positive tool-call count, or a run that finishes before the boundary is a
clear error instead of a partial checkpoint.

## Task 4: Restore a Fresh Model

`resume_agent(checkpoint, fresh_generate, workspace, limits=None)` validates the
checkpoint, calls:

```python
fresh_generate.restore_checkpoint(checkpoint.model)
```

rebuilds the semantic message list, and enters the normal loop. The fresh model
therefore sees the exact conversation prefix and produces the response at the
saved `response_index`.

No old tool action is submitted again. The first model call after restore sees
the already-recorded tool result and chooses the next action or final answer.

## Task 5: Make the Fake Cache Visible

The test's `FakeCheckpointModel` turns each message content length into one fake
token ID. `save_checkpoint(messages)` records those token IDs, two matching
layer offsets, and the next scripted-response index. A new model object restores
that state and asserts that its first resumed input has the same token prefix
and conversation position.

This deliberately small representation exposes the inference/harness
integration: the harness owns semantic conversation state, while the model owns
the cache snapshot that accelerates exactly that state.

## Task 6: Resume Without Replaying Effects

The end-to-end test scripts:

```text
read app.py
edit app.py
run the exact validation command
checkpoint after the validation observation
construct a fresh scripted model
resume to the final answer
```

Before resume, the Day 3 store already contains the edit receipt and validation
receipt. After resume, the approval log and receipt count are unchanged: neither
completed effect ran twice. The checkpoint contains no receipt IDs and makes no
exactly-once claim; it simply resumes after the conversation already says those
effects completed.

## Task 7: Keep the Boundary Small

The Day 4 starter adds only `checkpoint.py` and two loop entry points. Do not add
session IDs, parent pointers, branches, rewind methods, compaction summaries,
steering queues, disk cache files, or later-day modules. If the in-memory
checkpoint is lost, start a new run.

## Checkpoint

You can now stop after a complete tool observation and continue with a fresh
scripted model from the same conversation and fake-cache position. Inspect the
checkpoint's messages and model fields, then confirm that the pre-checkpoint
edit and command remain single completed effects.

Continue with [Day 5: Compact Completed Work](week4-05-compaction.md) to derive
a smaller model-visible transcript while keeping the exact effect receipts.

{{#include copyright.md}}
