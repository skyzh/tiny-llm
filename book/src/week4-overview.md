# 🚧 Week 4: Build a Coding Agent

> **Course status:** Week 4 is being published one checkpoint at a time. Day 1
> is ready to learn and review. Later days are deliberately not in this book
> yet; each will appear only after its implementation, starter, and reviews are
> ready.

Weeks 1 through 3 turn tokens into text and make serving that text efficient.
Week 4 starts a different kind of program: an agent repeatedly asks the model
for one structured action, records the result, and gives that result back to
the model. Today, that program is a bounded, deterministic loop. It does not
yet run a command, edit a file, save a session, or evaluate a task.

## What Day 1 Builds

Day 1 establishes the protocol and control flow that later checkpoints will
extend:

```text
task + system instruction
          |
          v
     model response
          |
          v
  validate one JSON action
      |             |
      | tool        | final answer
      v             v
 execute through  stop the run
 a supplied workspace
      |
      v
 append an observation and continue
```

The workspace in the Day 1 tests is a small fake object with one enabled
read-only action. This keeps the learning target focused: validate the model's
text, bound the loop, and preserve the conversation. The real effect boundary
will arrive in a later, separately reviewed checkpoint.

## Day 1 Checkpoint

The Day 1 starter exposes exactly these public names:

| File | Public names | Why they are here |
| --- | --- | --- |
| `src/tiny_llm/agent/generation.py` | `GenerationStats`, `initial_messages`, `generate_response` | Begin a conversation and keep the one-response model boundary explicit. |
| `src/tiny_llm/agent/protocol.py` | `AgentError`, `FinalAction`, `ToolAction`, `TOOL_CATALOG_HASH`, `tool_catalog_hash`, `parse_action`, `build_system_prompt` | Represent and validate one final answer or one tool request. |
| `src/tiny_llm/agent/loop.py` | `AgentLimits`, `AgentEvent`, `AgentRun`, `run_agent` | Bound the run and retain an inspectable trace. |

Implement the Day 1 exercise, then run:

```bash
pdm run test --week 4 --day 1
```

The learner test uses scripted responses and a fake workspace, so it does not
download or load a model. To check the supplied implementation without copying
the learner test, run:

```bash
pdm run test-refsol --week 4 --day 1
```

## Seven-Day Shape

The course will keep this seven-day structure. Only the first row is available
today; the rest is a map, not an invitation to implement ahead.

| Day | Theme | Availability |
| --- | --- | --- |
| 1 | Validated agent loop and tool protocol | Available now |
| 2 | Effect receipts | Published after its own reviewed merge |
| 3 | Session tree | Published after its own reviewed merge |
| 4 | KV checkpoint and rewind | Published after its own reviewed merge |
| 5 | Receipt-backed compaction | Published after its own reviewed merge |
| 6 | Steering, status, and reconciliation | Published after its own reviewed merge |
| 7 | Equivalence harness | Published after its own reviewed merge |

Do not add modules for a later day to your Day 1 solution. The starter's small
surface is intentional: it lets the tests catch accidental solution leakage
and makes each new capability easy to review.

{{#include copyright.md}}
