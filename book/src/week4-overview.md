# 🚧 Week 4: Build a Coding Agent

> **Course status:** Week 4 is being published one checkpoint at a time. Days 1
> through 6 are ready to learn and review. Additional capabilities will appear
> only after their implementation, starter, and reviews are ready.

Weeks 1 through 3 turn tokens into text and make serving that text efficient.
Week 4 starts a different kind of program: an agent repeatedly asks the model
for one structured action, records the result, and gives that result back to
the model. Day 1 builds the bounded deterministic loop. Day 2 gives that loop a
small read-only workspace for listing and reading project files. Day 3 adds an
approval boundary for edits, one exact validation command, and simple receipts.
Day 4 saves a complete conversation boundary and the fake model's cache
metadata, then restores both into a fresh model without replaying effects.
Day 5 derives a smaller model-visible transcript from older completed effects
while their exact receipts remain unchanged. Day 6 inspects one safe checkpoint,
adds one visible operator steering message, and resumes a fresh model without
replaying the completed effect.

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
text, bound the loop, and preserve the conversation. Day 2 replaces the fake
with real read-only tools without changing the Day 1 loop contract.

## Day 1 Checkpoint

The Day 1 starter exposes exactly these public names:

| File | Public names | Why they are here |
| --- | --- | --- |
| `src/tiny_llm/agent/generation.py` | `initial_messages`, `generate_response` | Begin a conversation and keep the one-response model boundary explicit. |
| `src/tiny_llm/agent/protocol.py` | `AgentError`, `FinalAction`, `ToolAction`, `parse_action`, `build_system_prompt` | Represent and validate one final answer or one tool request. |
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

## Publication Boundary

After Day 1 passes, continue with [Day 2: Inspect a
Workspace](week4-02-tools.md). The cumulative Day 2 command is:

```bash
pdm run test --week 4 --day 2
```

After Day 2 passes, continue with [Day 3: Edit, Validate, and
Record](week4-03-safe-editing.md). Its cumulative command is:

```bash
pdm run test --week 4 --day 3
```

After Day 3 passes, continue with [Day 4: Checkpoint and
Resume](week4-04-sessions.md). Its cumulative command is:

```bash
pdm run test --week 4 --day 4
```

After Day 4 passes, continue with [Day 5: Compact Completed
Work](week4-05-compaction.md). Its cumulative command is:

```bash
pdm run test --week 4 --day 5
```

After Day 5 passes, continue with [Day 6: Inspect and Steer a Paused
Agent](week4-06-steering.md). Its cumulative command is:

```bash
pdm run test --week 4 --day 6
```

Only the Day 1 through Day 6 starter modules are published. Do not add session
trees, rewind, reconciliation, evaluation, or other later public APIs to your
solution.

{{#include copyright.md}}
