# 🚧 Week 4: Stateful Inference for an Interactive Agent

> 🚧 **Course status:** The daily chapters are design drafts and are not
> included in the rendered book yet.

Weeks 1 through 3 built a model, optimized its operators, and put requests into
a paged continuous-batching engine. Week 4 asks what happens after a request
finishes. An interactive application sends another turn whose prompt is mostly
the conversation the engine just computed. Throwing that KV state away is
correct, but unnecessarily expensive.

A small coding agent supplies the workload. It alternates between model turns
and slow tools, accepts steering, creates checkpoints, and resumes after process
exit. Those behaviors force coordinated changes across prompt rendering,
generation, cache ownership, the page allocator, persistence, and scheduling.
The agent is not a separate application week; it is the integration test for the
inference framework built so far.

## The Baseline and the New Boundary

The starting API intentionally hides inference state:

```python
Generate = Callable[[list[Message]], str]

def run_agent(task: str, generate: Generate, workspace: Workspace) -> AgentRun:
    ...
```

`generate_response()` renders all messages, allocates one fresh cache per model
layer, prefills the complete prompt, decodes one action, and releases every
cache. The agent owns conversation events while the model adapter owns a single
stateless call. There is no token history, checkpoint, or resume contract.

During Week 4, replace that callback with an explicit session:

```python
class InferenceSession:
    def generate(self, messages: list[Message], max_tokens: int) -> Generation: ...
    def checkpoint(self) -> CacheCheckpoint: ...
    def restore(self, checkpoint: CacheCheckpoint) -> None: ...
    def save(self, path: Path) -> None: ...
    def close(self) -> None: ...
```

The agent still passes semantic messages rather than K/V tensors. The session
renders and tokenizes them, reconciles the new token sequence with its cached
sequence, and exposes lifecycle operations. Cache implementations remain below
that boundary.

## Three Kinds of State

Keep three similar-looking operations separate:

1. **Conversation state** is the durable event log and remains the source of
   truth after a cache miss or incompatible snapshot.
2. **Inference state** contains rendered token IDs and per-layer K/V. It is a
   disposable acceleration artifact tied to one model and cache layout.
3. **Workspace state** contains file mutations. Rewinding the conversation must
   not silently overwrite files, and undoing a file must not erase the trace.

A disk cache snapshot is therefore never the only copy of a session. Restore
must validate a model/configuration fingerprint, tokenizer and chat-template
fingerprint, dtype, page size, tensor shapes, logical lengths, and format
version. On any mismatch, discard it and replay the durable conversation.

## Cache Reconciliation

The next rendered prompt is often close to, but not byte-for-byte appended to,
the prior prompt. Adding an assistant message can replace a generation marker;
steering and compaction can change an earlier suffix. The session compares token
IDs, finds their longest common prefix, rewinds each layer to that boundary, and
prefills only the new suffix:

```text
cached:  [system | task | action A | generation marker]
next:    [system | task | action A | tool result | generation marker]
                         ^ keep ^     ^ prefill this suffix
```

Correctness is tested against a cold full prefill. Prefix reuse is an
optimization, so every warm path must produce the same logits and generated
tokens as the stateless baseline.

## Seven-Day Plan

| Day | Topic | Inference-framework milestone |
| --- | --- | --- |
| 1 | Interactive baseline | Trace prompt tokens, prefill work, latency, and page use across agent turns. |
| 2 | Stateful generation | Reuse an exact prompt prefix through a session-owned cache. |
| 3 | Reconciliation and rewind | Find token-level divergence and transactionally truncate dense and paged caches. |
| 4 | Checkpoint, fork, and restore | Share full prefix pages and copy a partial tail page only when a branch writes. |
| 5 | Durable warm state | Save a compact, versioned snapshot and safely fall back to transcript replay. |
| 6 | Session scheduling | Pause, resume, batch, evict, and restore interactive sessions under a page budget. |
| 7 | Agent-serving capstone | Preserve agent correctness while measuring warm-turn latency and cache efficiency. |

The supplied agent harness retains a deliberately small bounded tool surface:
list/read files, make exact edits, and run explicitly allowed commands. Its
safety policy remains important, but the learner-owned work this week is the
stateful inference path underneath it.

## Performance Questions

Every optimization has a regime where it loses. The capstone should answer:

- How many prompt tokens does a warm turn avoid prefilling?
- When is loading K/V from disk slower than replaying the transcript?
- How much memory does copy-on-write save across two branches?
- Does keeping paused sessions resident hurt active decode throughput?
- Which eviction policy improves time to first token under a fixed page budget?
- Do warm and cold execution remain token-for-token equivalent?

This continues the measurement discipline from Weeks 2 and 3 while requiring
students to optimize across components rather than inside one kernel.

## Run the Starting Baseline

```bash
pdm run agent "inspect this project and summarize its files"
```

The first run uses the current stateless adapter. Day 1 adds an interactive mode
and metrics without changing its behavior. Later checkpoints select the
stateful engine, cache policy, and persistence directory explicitly so cold and
warm runs remain easy to compare.

{{#include copyright.md}}
