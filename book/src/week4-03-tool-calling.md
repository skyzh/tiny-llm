# 🚧 Week 4: Tool Calling and Agent Serving

> 🚧 This chapter is under review and may change.

This chapter treats tool calling as an API boundary and then places that boundary
on top of the Week 3 serving runtime. The model proposes actions; trusted code
validates them, applies policy, executes one tool, and records the result.

## Objectives

By the end of this chapter, you should be able to:

- validate tool names and arguments before execution;
- expose only capabilities enabled by the operator;
- return structured observations and recoverable errors to the model;
- bound a tool-calling session by steps, tokens, context, and failures; and
- keep agent history, durable session state, and model KV-cache ownership
  separate when serving concurrent sessions.

## Prerequisites

- Complete the structured-action and workspace-policy tasks from the coding-agent
  chapter.
- Complete Week 3 continuous batching before adding concurrent sessions.
- Complete paged KV-cache ownership before experimenting with cache eviction or
  reuse across scheduling steps.

First validate a single session. Concurrency should not be used to hide an
incorrect action or state model.

## Task 1: Freeze the Tool Protocol

Define a small schema containing a tool name and that tool's arguments. Parse one
action at a time and reject malformed payloads, unknown tools, missing fields,
extra fields, and arguments of the wrong type.

Checkpoint: invalid model output produces a structured error and no tool side
effect.

## Task 2: Apply Policy Before Execution

Build the model's tool description from the active policy. Validate paths,
mutation permissions, command allowlists, output limits, and timeouts before
dispatch. A disabled capability must be absent from the prompt and rejected if
the model requests it anyway.

Checkpoint: the same proposed action succeeds under an enabling policy and fails
without side effects under a restrictive policy.

## Task 3: Return Structured Observations

Record the proposed action, validation result, tool output, and error category as
one complete turn. Feed recoverable errors back to the model so it can choose a
different action. Stop on a final answer, an exhausted budget, or repeated
failures.

Checkpoint: replaying the recorded turns explains every side effect and produces
the same next model context.

## Task 4: Add Serving State

Represent each live agent session with three distinct kinds of state:

1. durable task and event history;
2. scheduler-visible request state; and
3. model KV-cache state owned by the selected Week 3 cache implementation.

Do not store tool policy or durable history inside the KV cache. When a request
is retried, evicted, or resumed, update each state machine explicitly. Reuse the
Week 3 scheduler and cache interfaces rather than changing model kernels for an
application concern.

Checkpoint: two concurrent sessions can interleave model calls without sharing
messages, tool permissions, cache handles, or modified-path records.

## Validate the Chapter Checkpoint

Test the protocol and policy with malformed JSON, unknown tools, disabled tools,
path escapes, command timeouts, oversized output, and repeated invalid actions.
Then exercise two sessions whose requests finish at different times and verify:

- each observation returns to the session that issued the action;
- finishing or evicting one session releases only its cache;
- retrying a tool does not duplicate an already committed mutation;
- durable history survives independently of device-cache eviction; and
- aggregate serving metrics remain separate from single-request token latency.

Report valid-action rate, recovery rate, concurrent-session throughput, cache
reuse, and end-to-end task latency. These application metrics complement, but do
not replace, the kernel throughput measurements from Weeks 2 and 3.

{{#include copyright.md}}
