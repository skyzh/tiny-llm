# 🚧 Week 4: Tool Calling and Agent Serving

> 🚧 This chapter is a work in progress.

Tool calling turns model output into requests for external actions. Treat the
boundary as an API, not as free-form text:

- define a small schema for tool name and arguments;
- reject unknown tools and malformed arguments;
- enforce workspace and resource limits before execution;
- return structured observations to the model;
- stop on a final answer, a budget, or repeated failures.

Serving an agent adds request state that lasts across many model calls. Reuse
the Week 3 scheduler and cache abstractions, but keep durable agent state,
conversation history, and model KV state separate. This makes retries and cache
eviction explicit instead of coupling them to the tool executor.

{{#include copyright.md}}
