# Week 4: Build a Tiny Coding Agent

Weeks 1 through 3 turned tokens into text, made decoding efficient, and introduced
advanced serving techniques. Week 4 adds the next layer: an agent loop that lets
the model observe a workspace, choose a tool, see the result, and continue until
a coding task is complete.

The goal is not to reproduce a production coding agent. It is to understand the
small mechanism underneath one and identify where reliability and safety come
from. By the end of the week, you will have a local CLI agent powered by the
inference stack you already built.

## The Core Loop

1. Render the task and tool descriptions with the chat template.
2. Decode one structured action using the Week 2 model and KV cache.
3. Parse and validate the action.
4. Run one workspace tool and append its result to the conversation.
5. Repeat until the model returns a final answer or reaches a step limit.

The initial demo deliberately has only three tools: `list_files`, `read_file`,
and `write_file`. Small tool surfaces make behavior easier to inspect, test, and
secure before adding command execution.

## Seven-Day Plan

WIP

<!--
### Day 1: From Generation to an Agent Loop

Return generated text from the decoder, define an action protocol, and alternate
between model inference and tool execution. Trace one task by hand so every token,
action, and observation is visible.

### Day 2: Structured Tool Calling

Replace ad-hoc parsing with an explicit schema. Study malformed JSON, missing
arguments, unknown tools, retries, stop conditions, and how decoding constraints
could improve correctness.

### Day 3: Workspace Tools and Safety

Add path normalization, workspace boundaries, file-size limits, and clear error
messages. Discuss why shell execution, symlinks, secrets, and destructive writes
need stronger policies than file reads.

### Day 4: Context and State

Measure prompt growth across steps. Add truncation or summarization, separate
durable task state from chat history, and compare the full and paged KV caches
from earlier weeks for longer agent sessions.

### Day 5: Editing and Validation

Move from whole-file writes to patches, then add a narrowly scoped test or lint
tool. Teach the agent to inspect before editing and validate after editing.

### Day 6: Planning and Recovery

Introduce a lightweight plan, tool-error recovery, step and token budgets, and
loop detection. Compare direct action with plan-then-act behavior on multi-file
tasks.

### Day 7: Evaluation and Capstone

Build a small suite of repository tasks with expected files, tests, and forbidden
side effects. Report completion rate, malformed-action rate, steps, tokens, and
latency. The capstone is a safe agent that fixes a small bug and runs its test.

## Run the Demo

Complete the Week 2 exercises first, or use the included reference solution:

```bash
pdm run agent "inspect this project and add a hello.py example"
```

The demo defaults to `tiny_llm_ref`. Pass `--solution tiny_llm` to exercise your
implementation, or select `--loader week3` to use the paged KV cache. It also
supports the course's `--device`, `--enable-flash-attn`, and `--enable-thinking`
options. Run it from the workspace you want it to edit, and use `--max-steps` and
`--max-tokens` to cap work. The default Qwen3 4B model follows the JSON tool
protocol more reliably; use `--model qwen3-0.6b` on memory-constrained machines.

For a faster optimized executor, use MLX-LM directly. This keeps the same agent
loop and tools but bypasses the educational Week 2/3 model and KV cache:

```bash
pdm run agent --solution mlx "inspect this project and summarize its files"
```

`--loader` and `--enable-flash-attn` are ignored in this mode because MLX-LM
selects and manages its own optimized attention and cache implementations.

## Suggested Milestones

- Minimal: the model can list, read, and create a file without leaving the root.
- Reliable: invalid actions produce useful feedback and the agent can recover.
- Safe: writes are reviewable, command execution is allowlisted, and budgets bind.
- Measurable: a repeatable task suite distinguishes progress from anecdotes.
-->

{{#include copyright.md}}
