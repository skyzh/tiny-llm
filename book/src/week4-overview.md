# 🚧 Week 4: Build a Coding Agent

> **Course status:** Week 4 is being published one checkpoint at a time. Days 1
> through 9 are ready to learn and review. Additional capabilities will appear
> only after their implementation, starter, and reviews are ready.

Weeks 1 through 3 built the inference path: tokenize a conversation, run the
model, and manage the key/value cache that makes later decoding efficient.
Week 4 puts that path inside a small coding-agent harness. The model proposes
one structured action at a time; ordinary Python validates it, executes it
through an explicit workspace boundary, returns the observation, and decides
when the run must stop.

The goal is not broad autonomy. Each day adds one visible mechanism that you
can inspect and test: bounded control flow, read-only tools, approved effects,
receipts, checkpoints, context selection, steering, evaluation, prefix reuse,
and bounded evidence retrieval.

## The Nine-Day Progression

| Day | Question | Visible mechanism |
| --- | --- | --- |
| [1](week4-01-agent-loop.md) | How does model text become one safe next step? | A validated JSON action protocol and bounded loop. |
| [2](week4-02-tools.md) | How can the agent inspect a project? | A contained read-only workspace with listing and file reads. |
| [3](week4-03-safe-editing.md) | How can it change and validate code deliberately? | Operator approval, exact edits and commands, and effect receipts. |
| [4](week4-04-sessions.md) | Where can a run pause and resume? | One complete-observation checkpoint with model cache metadata. |
| [5](week4-05-compaction.md) | How can older completed work use less context? | Receipt-backed deterministic compaction of completed interactions. |
| [6](week4-06-steering.md) | How can an operator inspect and redirect a paused run? | A public status view and one visible steering message. |
| [7](week4-07-evaluation.md) | How do we judge a run without grading hidden reasoning? | A deterministic report over declared observable outcomes. |
| [8](week4-08-fork-steer-select.md) | How can two continuations reuse one inference prefix? | Dense tokenizer/KV-prefix reuse, isolated effects, and explicit selection. |
| [9](week4-09-bound-tool-evidence.md) | How can a large tool result remain available without filling the prompt? | Content-addressed external bytes, bounded previews, and explicit range retrieval. |

The sequence is cumulative. Later starters keep the earlier public surface, and
later tests rely on the boundaries established before them. Work through the
days in order even when one later mechanism is your main interest.

## Prerequisites and Environment

Complete the repository setup and Weeks 1 through 3 first. Day 8 directly uses
the course tokenizer, model, and dense KV cache; the earlier Week 4 days also
assume you recognize the model-generation boundary those weeks established.

The supported environment is macOS on Apple Silicon with the project
dependencies installed. The deterministic Week 4 learner tests use scripted
models and temporary workspaces, so they do not need model downloads. The
chapters that include real-model walkthroughs label them as manual and
nondeterministic; uncached runs additionally need network access, disk space,
and enough Apple unified memory for the selected MLX weights.

Use only disposable workspaces with no secrets. Read observations become model
input, and later days can enable file changes and an exact allowlisted command.

## How to Use This Week

For each day:

1. Read its teaching boundary and public starter surface.
2. Run the day-specific learner command to see the expected failures.
3. Implement only that day's numbered tasks in `src/tiny_llm/agent/`.
4. Rerun the same cumulative checkpoint until it is green.
5. Inspect the recorded events, files, receipts, reports, or cache facts named
   by the chapter instead of trusting a final model sentence alone.

The day chapters contain the exact commands and checkpoints. Course maintainers
can use the corresponding reference command without copying a learner test.
Optional real-model walkthroughs come after the deterministic checkpoint; they
are exploration, not a replacement for reproducible course-code evidence.

## The Cumulative Learning Arc

Days 1 through 3 establish the ordinary agent cycle: propose, validate,
observe, approve effects, and record evidence. Days 4 through 6 show how the
harness can pause, reduce older context, and add an operator instruction while
preserving completed work. Day 7 evaluates observable outcomes. Day 8 reconnects
that control path to the tokenizer and KV cache from Weeks 1 through 3. Day 9
keeps oversized results verifiable while exposing only a bounded view to the
model.

By the end, you can explain both sides of the boundary: what the model sees and
proposes, and what the harness validates, executes, retains, or refuses.

## Week Boundary

This is a teaching agent for a trusted operator and disposable local projects.
It is not a sandbox, hostile-filesystem defense, process jail, durable
transaction system, distributed scheduler, session tree, hidden grader,
semantic-perfect memory, network artifact service, or production serving
framework. Completed effects are never presented as rewound, and a model's
final prose is never treated as proof by itself.

Each day states its narrower limits where they matter. Keep later or
production-scale APIs out of the starter unless a chapter explicitly teaches
them.

{{#include copyright.md}}
