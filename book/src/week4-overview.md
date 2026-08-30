# 🚧 Week 4: Build a Coding Agent

Weeks 1 through 3 ended with a working inference path: render a conversation,
run the model, and carry its KV cache into later decoding. Week 4 asks what has
to surround that path before model text can act on a project.

The product you are building is one bounded coding-agent run:

```text
task -> model response -> validated action -> workspace observation
     -> approved effect -> receipt -> checkpoint -> compacted view
     -> visible steering -> observable evaluation
     -> two isolated continuations -> explicit selection
     -> bounded retrieval of oversized evidence
```

Each arrow is a harness decision, not a model privilege. The model proposes one
JSON action. Ordinary Python decides whether the action is well formed,
enabled, approved, executed, retained, or refused.

## What Is Runnable Today

The repository currently ships one **cumulative Day 9 declaration scaffold**.
All Week 4 modules and exports are visible from Day 1, but later-day
implementation surfaces remain out of scope until their chapter. Most are
TODO stubs; Day 9 explicitly supplies one constructor-validation rule. This is
not nine separately materialized starters.

The deterministic learner checkpoint for each chapter is also currently
**day-local**: `pdm run test --week 4 --day N` runs the copied Day N test, not
Days 1 through N. A cumulative runner and one supplied Days 1--9 capstone are
still pending. Until those arrive, rerun earlier day tests when changing a
shared surface; do not treat a green later-day file as proof that every prior
mechanism still works.

The real-model `pdm run agent` command currently exercises the learner loop,
workspace, approvals, and receipts from Days 1--3, but its MLX-LM adapter calls
`mlx_lm.generate` directly. It does **not** exercise the learner-owned
`generate_response` helper or the Week 1--3 course model/cache path. Day 8
reconnects to that path in a deterministic test and a manual walkthrough. The
final composed product command remains pending with the cumulative capstone.

These limits are visible course state, not goals for the learner to repair in
the prose-only checkpoint.

## The Nine-Day Progression

| Day | Product pressure | Learner-owned mechanism | Evidence to inspect |
| --- | --- | --- | --- |
| [1](week4-01-agent-loop.md) | Model text is not yet a safe next step. | A validated JSON action protocol and bounded loop. | Parsed events, exact observations, and stop reasons. |
| [2](week4-02-tools.md) | A fake workspace cannot inspect a project. | Contained directory listing and UTF-8 reads. | Listed paths, returned bytes, and recoverable errors. |
| [3](week4-03-safe-editing.md) | A read-only agent cannot finish a coding task. | Approval, exact edits and commands, and effect receipts. | Changed bytes, validation status, and receipt facts. |
| [4](week4-04-sessions.md) | A stopped process loses its conversation/model position. | One complete-observation checkpoint and resume boundary. | Saved messages/cache metadata and no effect replay. |
| [5](week4-05-compaction.md) | Completed evidence consumes prompt space. | Receipt-backed deterministic compaction. | Tokens before/after, saved tokens, and unchanged receipts. |
| [6](week4-06-steering.md) | An operator needs a visible correction point. | Inspect, append one steering message, and resume. | Public status and message ordering. |
| [7](week4-07-evaluation.md) | A final sentence is not proof. | A report over declared observable outcomes. | Named file/result/receipt checks. |
| [8](week4-08-fork-steer-select.md) | Two continuations should not prefill one identical prefix twice. | Dense token/KV-prefix reuse, isolated effects, and explicit selection. | Prefix offsets, avoided logical prefill, branch-local facts, and reports. |
| [9](week4-09-bound-tool-evidence.md) | A large result should not fill every later prompt. | Content-addressed bytes, bounded previews, and exact range retrieval. | Artifact size/digest, omitted interval, and returned range. |

The mechanisms are designed to compose in that order. Today, Days 4--9 are
exercised as library checkpoints rather than one runnable CLI product. The
pending capstone will supply the orchestration shell; it will not replace the
mechanisms you implement here.

## Prerequisites and Environment

Complete repository setup and Weeks 1 through 3 first. Day 8 directly uses the
course tokenizer, model, and dense KV cache. The other deterministic Week 4
tests use scripted models and temporary workspaces, so they need no model
download.

The supported native environment is macOS on Apple Silicon with the project
dependencies installed. Real-model sections are manual and nondeterministic.
An uncached run also needs network access, free disk space, and enough unified
memory for the selected MLX weights. Use only disposable workspaces with no
secrets: tool observations become model input, and Day 3 can enable file
changes plus one exact allowlisted command.

## Work Through One Chapter

For Day N:

1. Read what the final scaffold already declares and which TODO bodies belong
   to this day. Ignore future modules even though their declarations are
   visible.
2. Predict the named action, count, range, or stop reason before running the
   focused scenario when the chapter asks for one.
3. Copy the supplied Day N test explicitly, then run the day-local checkpoint:

   ```bash
   pdm run copy-test --week 4 --day N
   pdm run test --week 4 --day N
   ```

   `copy-test` refreshes the learner copy from the supplied checkpoint. If you
   skip it, `pdm run test` copies only when the target is absent; a differing
   existing copy is preserved with a warning.
4. Implement only the files and relationships named by that chapter.
5. Rerun the checkpoint and inspect the artifact that can falsify your
   prediction: events, files, receipts, checkpoints, reports, cache offsets, or
   artifact bytes.
6. When you change shared code, rerun the earlier affected day tests manually
   until the cumulative runner is available.

Course maintainers can run the corresponding `test-refsol` command without
copying a learner test. Optional model walkthroughs come after the deterministic
checkpoint; they are exploration, not correctness evidence.

## Read the Metrics as Accounting

Week 4 exposes three kinds of useful counts:

- Day 5: transcript tokens before and after compaction;
- Day 8: reused prefix tokens, layer offsets, and avoided-prefill tokens; and
- Day 9: complete artifact bytes, model-visible bytes, and returned range
  bytes.

These values prove identity and logical-work accounting inside the teaching
mechanisms. They do not establish wall-clock speedup, throughput, model
quality, memory-capacity gain, or a universal policy. A manual cached-Qwen run
may record the model ID, cache state, device, and observed actions, but its
choices remain nondeterministic and non-comparative.

## Week Boundary

This is a teaching agent for a trusted operator and disposable local projects.
It is not a sandbox, hostile-filesystem defense, process jail, durable
transaction system, distributed scheduler, session tree, hidden grader,
semantic-perfect memory, network artifact service, or production serving
framework. Completed effects are never presented as rewound, and a model's
final prose is never treated as proof by itself.

{{#include copyright.md}}
