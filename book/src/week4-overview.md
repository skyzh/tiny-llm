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

The deterministic learner checkpoint is cumulative within Week 4:
`pdm run test --week 4 --day N` force-refreshes the supplied learner tests for
Days 1 through N, then runs those files together. A later checkpoint therefore
rechecks every earlier mechanism it builds on.

After repository setup, start with the [Day 2 fixed fixture](week4-02-tools.md#inspect-the-supplied-fixture).
It runs the reference loop offline in a temporary workspace and checks the
approved Day 3 greeting edit, notes write and validation.

The interactive `pdm run agent` command uses the Week 2 model,
`generate_response()` and a fresh growing cache per response. Day 2 shows
how to use a cached Qwen3 model with read-only access to the fixed sample.
After all nine days are complete, `pdm run week4-capstone` exercises the
deterministic flow through sessions, branches and bounded evidence.

<a id="the-nine-day-progression"></a>

## Five Modules, Nine Days

The week teaches five ideas. Each is built over one or two days:

| Module | What it teaches | Days |
| --- | --- | --- |
| **Control** | Turn model text into one validated action, and stop at a limit | 1 |
| **Effects** | Read a workspace, then make approved edits and commands that leave receipts | 2, 3 |
| **Sessions** | Pause after a complete observation, inspect it, steer it and resume | 4, 6 |
| **Context** | Keep prompts small: compact completed work and move oversized evidence out of line | 5, 9 |
| **Branches** | Judge a run by what it observably did, then run two continuations and pick one explicitly | 7, 8 |

Work through the days in number order. Each day's check reruns every earlier
day, so the modules interleave: Day 6 steers the checkpoint you made on Day 4,
and Day 8 chooses a branch using the report you built on Day 7.

| Day | Module | Product pressure | Learner-owned mechanism | Evidence to inspect |
| --- | --- | --- | --- | --- |
| [1](week4-01-agent-loop.md) | Control | Model text is not yet a safe next step. | A validated JSON action protocol and bounded loop. | Parsed events, exact observations, and stop reasons. |
| [2](week4-02-tools.md) | Effects | A fake workspace cannot inspect a project. | Contained directory listing and UTF-8 reads. | Listed paths, returned bytes, and recoverable errors. |
| [3](week4-03-safe-editing.md) | Effects | A read-only agent cannot finish a coding task. | Approval, exact edits and commands, and effect receipts. | Changed bytes, validation status, and receipt facts. |
| [4](week4-04-sessions.md) | Sessions | A stopped process loses its conversation/model position. | One complete-observation checkpoint and resume boundary. | Saved messages/cache metadata and no effect replay. |
| [5](week4-05-compaction.md) | Context | Completed evidence consumes prompt space. | Receipt-backed deterministic compaction. | Tokens before/after, saved tokens, and unchanged receipts. |
| [6](week4-06-steering.md) | Sessions | An operator needs a visible correction point. | Inspect, append one steering message, and resume. | Public status and message ordering. |
| [7](week4-07-evaluation.md) | Branches | A final sentence is not proof. | A report over declared observable outcomes. | Named file/result/receipt checks. |
| [8](week4-08-fork-steer-select.md) | Branches | Two continuations should not prefill one identical prefix twice. | Dense token/KV-prefix reuse, isolated effects, and explicit selection. | Prefix offsets, avoided logical prefill, branch-local facts, and reports. |
| [9](week4-09-bound-tool-evidence.md) | Context | A large result should not fill every later prompt. | Content-addressed bytes, bounded previews, and exact range retrieval. | Artifact size/digest, omitted interval, and returned range. |

The mechanisms compose in that order. After Day 9, the
[live walkthrough](week4-live-walkthrough.md) runs all five modules with a
cached Qwen3-4B on the Week 2 engine. The supplied deterministic
capstone is the orchestration shell that exercises those completed APIs
together; it does not replace the mechanisms you implement here.

## Prerequisites and Environment

Complete repository setup and Weeks 1 through 3 first. Day 8 directly uses the
course tokenizer, model, and dense KV cache. The other deterministic Week 4
tests use scripted models and temporary workspaces, so they need no model
download.

The supported native environment is macOS on Apple Silicon with the project
dependencies installed. The interactive fixture uses a locally cached Qwen3
model. Download the weights during setup and budget disk space and unified
memory for the chosen size. Use disposable workspaces with public fixture
data for manual runs. Tool observations become model input, and Day 3 can
enable approved file effects and exact commands.

## Work Through One Chapter

For Day N:

1. Read what the final scaffold already declares and which TODO bodies belong
   to this day. Ignore future modules even though their declarations are
   visible.
2. Predict the named action, count, range, or stop reason before running the
   focused scenario when the chapter asks for one.
3. Run the cumulative learner checkpoint:

   ```bash
   pdm run test --week 4 --day N
   ```

   For Week 4, this command force-refreshes the learner tests for Days 1
   through N from the supplied checkpoints, then runs all of them in one
   pytest invocation. Keep your implementation in `src/`; do not modify a
   copied test because the next run replaces it.
4. Implement only the files and relationships named by that chapter.
5. Rerun the checkpoint and inspect the artifact that can falsify your
   prediction: events, files, receipts, checkpoints, reports, cache offsets, or
   artifact bytes.
6. After Day 9 is green, run the composed product witness:

   ```bash
   pdm run week4-capstone
   ```

   Inspect its sorted JSON sections for compaction, both branches, the selected
   branch, and the externalized artifact range.

Course maintainers can run the corresponding day-local `test-refsol` command
without copying learner tests. Optional model walkthroughs come after the
deterministic checkpoint; they are exploration, not correctness evidence.

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
