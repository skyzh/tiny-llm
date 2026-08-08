# Day 4: Interactive Sessions and Resume

> 🚧 **Early-review WIP:** This chapter is public for early review and may
> change. Use a disposable workspace when running the agent or enabling writes
> or commands.

The stateless loop from the first three days loses its conversation whenever the
process exits. This chapter makes the event stream durable and adds a reusable
generation cache without making either representation depend on the other.

> **Implementation status:** The reference implementation, learner API surface,
> and focused tests in this chapter are executable. The chapter remains WIP even
> though the checkpoint is executable.

## Check the Chapter

Implement the session and generation APIs under `src/tiny_llm/agent/`, then run:

```bash
pdm run test --week 4 --day 4
```

Use `pdm run test-refsol --week 4 --day 4` to check the supplied implementation.
The tests use temporary directories and fake caches; they do not execute model
commands or destructive subprocesses.

## Three Separate States

The implementation deliberately separates:

1. **Durable session state:** append-only user, assistant, tool, and lifecycle
   events.
2. **Model-visible context:** semantic chat messages rebuilt from those events.
3. **KV state:** an in-process optimization for one rendered token prefix.

The JSONL log is canonical. Losing or closing a KV cache only makes the next
turn perform a cold prefill; it does not make the conversation impossible to
resume.

## Append-Only Session Log

`SessionEvent` stores an ID, UTC timestamp, event type, optional parent ID, and
type-specific data. `SessionLog.append()` serializes ID assignment and append,
flushes the line, and calls `fsync` before returning. A persisted session begins
with metadata containing its resolved workspace, model/backend/template
identifier, and loaded project instructions.

The agent records events in this order:

```text
user_message
assistant_message        # durable before parsing or executing
tool_call
tool_result
run_finished
```

If a process exits after `tool_call`, resume appends a concise interrupted
`tool_result`. It never repeats the side effect or invents a successful result.
Malformed, oversized, duplicate-ID, symlinked, or metadata-inconsistent logs
fail closed.

Session transcripts can contain source text and command output. They live under
`.tiny-llm/sessions`, are ignored by Git, and are hidden from the model's
workspace tools. Treat that directory as sensitive local data.

## Resume Boundaries

`SessionStore` creates, loads, and selects the newest session. Loading validates
the resolved workspace and model/backend/template identifier. A session cannot
silently be replayed against a different repository or inference template.

At creation and resume, the store reads the workspace-root `AGENTS.md` when it
is a bounded, regular UTF-8 file. The snapshot and SHA-256 digest are recorded.
If the file changes, an `instructions_changed` event explains which policy is
now visible. Recursive instruction discovery and arbitrary configuration files
are intentionally deferred.

The CLI supports:

```text
agent TASK                 start and persist a new session
agent --interactive TASK   accept follow-up messages in the same process
agent --continue           resume the newest session for this workspace/model
agent --session ID         resume one selected session
agent --no-session TASK    run without creating a session file
```

`--continue` and `--session` are mutually exclusive. `--no-session` cannot be
combined with resume flags. Existing Day 3 safety rules still apply to every
tool call after resume. A completed session needs a new interactive follow-up;
resume never creates an unsolicited model turn.

## Rebuilding Context

The context builder maps user and assistant events back to their chat roles and
wraps completed tool results as user observations. Lifecycle, timing, and
metadata events remain audit-only. A recovered unmatched call becomes a normal
error observation, so the next model turn can choose a safe next action.

Day 4 still uses the character-bounded retention helper once the conversation
is rebuilt. Token-aware reduction and structured summaries belong to Day 5;
the durable log is never trimmed.

## Reusing KV State

`GenerationSession` preserves the existing one-argument `Generate` boundary:

```python
response = generation_session(messages)
```

It renders and tokenizes the requested messages, compares token IDs with the
prefix represented by every live cache layer, and computes their longest common
prefix. Before rewinding, it verifies that all layer offsets agree with its own
bookkeeping. A valid divergent suffix is rewound on every layer and only the new
suffix is prefetched. Any disagreement, unsupported rewind, or cache error
releases the whole cache set and starts cold.

`GenerationStats` reports input, reused, rewound, prefetched, and output token
counts plus whether the turn started cold. The represented-token invariant is:

```text
every cache offset == len(the token IDs represented by the live cache)
```

The final sampled token is included only after it has actually been fed through
the model. `close()` is idempotent and releases every cache. Warm and cold paths
must produce the same greedy response.

The reusable cache is implemented for the course Week 2/3 model adapter. The
`--solution mlx` compatibility backend keeps its stateless generator; durable
session replay still works there.

## Exercise

1. Round-trip every session event through JSONL.
2. Verify that an assistant response is durable before a tool runs.
3. Resume an unmatched tool call without re-executing it.
4. Reject a workspace or model mismatch.
5. Change `AGENTS.md` and inspect the recorded transition.
6. Confirm that `--no-session` writes nothing.
7. Compare cache reuse, rewind, and cold-fallback counters with fake layers.
8. Close a live generation session twice without leaking a cache.

Day 5 adds rendered-token budgeting and structured compaction. Day 6 adds
mutation recovery, checkpoints, cancellation, steering, and branches. Disk KV
snapshots and multi-process session writers are not part of this checkpoint.

{{#include copyright.md}}
