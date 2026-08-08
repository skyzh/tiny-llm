# Day 5: Token-Aware Context Compaction

> 🚧 **Early-review WIP:** This chapter is public for early review and may
> change. Use a disposable workspace when running the agent or enabling writes
> or commands.

An append-only session grows forever, but the model has a finite context window.
Day 5 derives a bounded model-visible working set without deleting or rewriting
the durable trace from Day 4.

> **Implementation status:** The reference implementation, learner API surface,
> and focused tests in this chapter are executable. The chapter remains WIP even
> though the checkpoint is executable.

## Check the Chapter

Implement the context APIs under `src/tiny_llm/agent/`, then run:

```bash
pdm run test --week 4 --day 5
```

Use `pdm run test-refsol --week 4 --day 5` for the supplied implementation. The
new compaction tests use synthetic transcripts and injected encoders/summarizers;
the retained Day 3 safety tests use mocked processes rather than executing agent
commands.

## The Executable Policy

`ContextPolicy` makes every budget explicit:

```python
ContextPolicy(
    max_tokens=32_768,
    reserve_tokens=8_192,
    summary_max_tokens=1_024,
    max_tool_result_tokens=4_096,
    min_recent_turns=2,
)
```

The working input limit is `max_tokens - reserve_tokens`, or 24,576 tokens for
the course Qwen3-4B default. The reserve covers the next response and observation;
it is not extra model context.

`ContextManager` receives the same exact message encoder used by generation:

```python
manager = ContextManager(generation.encode_messages, policy)
window = manager.prepare(session, system_prompt, summarize)
```

`ContextWindow.token_ids` therefore counts the fully rendered request, including
chat-template framing, the system tool schema, project instructions, messages,
summaries, and visible tool results. Character counts and per-message token
estimates are not accepted as the limit.

If the immutable anchors plus the minimum recent tail cannot fit, preparation
raises `ContextLimitError`. The loop stops once with `reason="context_limit"`;
it never drops the current request, repeatedly retries compaction, or calls the
model with a known-overflowing request.

## Bound Observations Before Summarizing

The durable `tool_result` event always retains its original bounded tool output.
Only its model-visible rendering may be reduced further:

- listings keep the useful head;
- command-style output keeps the useful tail; and
- file-like text keeps a head and tail separated by an omission marker.

The manager uses the exact encoder to verify the reduced result against
`max_tool_result_tokens`. This prevents one large observation from consuming the
entire compaction budget while preserving the canonical evidence for audit and
later evaluation.

## Structured Working Summary

Older complete events are replaced in model-visible context by one strict
`WorkingSummary`:

```json
{
  "goal": "Fix parsing of empty configuration values",
  "constraints": ["Do not change the public configuration schema"],
  "facts": ["parse_value is defined in src/config.py"],
  "changed_files": ["src/config.py"],
  "validation": ["test_empty_value still fails"],
  "failed_approaches": ["The first exact edit produced a tool error"],
  "next_step": "Inspect normalization before parse_value"
}
```

The parser requires exactly these keys, non-blank required strings, immutable
string tuples, and fixed item/size bounds. Extra keys and wrong types are
rejected. The original goal remains anchored. Even for an accepted model
summary, changed paths and command status are reconciled from successful
structured tool results rather than claims embedded in untrusted output text.

An optional summarizer receives a dedicated schema instruction and semantic
messages and may return this JSON once. The manager exact-encodes that request
and reserves the configured summary output before calling the model. Its raw
response or error is recorded in a bounded, audit-only `summary_attempt` event.
Invalid JSON, schema failure, an exception, or an over-budget request or summary
immediately selects the deterministic fallback; there is no recursive retry.

The CLI supplies a fresh temporary generation cache for summary work, so the
primary agent cache is unchanged until the validated compaction event is
durable. Tests and lightweight integrations may omit the callback and select the
deterministic strategy directly.

## Durable Compaction Marker

Compaction appends this event instead of changing older events:

```json
{
  "covered_through_event_id": "...",
  "strategy": "model",
  "fallback_reason": null,
  "summary": {"goal": "..."},
  "input_tokens_before": 25001,
  "input_tokens_after": 3812
}
```

The coverage boundary must reference an existing event and cannot cross an
unmatched tool call. Repeated compaction starts from the newest structured
summary plus later events; only the newest summary is model-visible. Original
JSONL events remain inspectable and replayable.

`ContextWindow` returns the exact token IDs, visible tool-result byte count,
whether this preparation appended a compaction, and that event's ID. The
assistant event records those context metrics. Course `GenerationSession`
backends also record `GenerationStats`, whose Day 5 fields now include latency;
the stateless MLX compatibility backend leaves generation metrics unknown.

## Cache Reconciliation

Compaction changes an older portion of the rendered prompt. The loop passes the
new semantic messages to Day 4's `GenerationSession`, which uses the same token
longest-common-prefix path as any other turn. Every layer is validated before a
rewind; compatible layers rewind the divergent suffix and prefill the new
summary/tail, while any inconsistent state is discarded for a cold prefill.

The event log and summary are semantic state. K/V tensors remain derived state
and are never summarized, edited, or required for restart correctness.

## Exercise

1. Build a synthetic session containing repeated reads, edits, and validation
   observations.
2. Measure its fully rendered token count.
3. Verify that under-budget preparation writes no compaction event.
4. Compact the old prefix and reload the JSONL session.
5. Force invalid summary JSON and inspect the deterministic fallback marker.
6. Confirm that the original tool result remains byte-for-byte unchanged.
7. Reconcile the warm generation cache and compare its response with a cold run.
8. Reduce the budget below the anchors and observe one fail-closed stop.

Day 6 adds write-ahead mutation recovery, checkpoints, undo, cancellation,
steering, and branches. RAG, disk K/V snapshots, adaptive cache heuristics, and
Day 7 grading remain explicit extensions.

{{#include copyright.md}}
