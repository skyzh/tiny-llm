# Week 4: Retrieval-Augmented Generation (WIP)

> This chapter is a work in progress.

RAG adds external context before generation. Build a small local pipeline:

1. split documents into inspectable chunks;
2. embed and index those chunks;
3. retrieve the top matches for a question;
4. render citations and retrieved text into the prompt;
5. generate an answer and record which chunks supported it.

Measure retrieval quality separately from generation quality. A fluent answer
cannot repair missing evidence, and a good retrieval result can still be lost
by poor prompt assembly or context truncation.

The capstone combines RAG with the coding agent: retrieve project conventions
or API notes, then expose the evidence to the tool-calling loop.

{{#include copyright.md}}
