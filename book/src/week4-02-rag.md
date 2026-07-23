# 🚧 Week 4: Retrieval-Augmented Generation

> 🚧 This chapter is under review and may change.

Retrieval-augmented generation adds selected external evidence to the prompt
before decoding. In this chapter, build a small local pipeline whose chunks,
retrieval results, citations, and prompt budget are all inspectable.

## Objectives

By the end of this chapter, you should be able to:

- split a document collection into stable, attributable chunks;
- embed and index those chunks;
- retrieve a bounded set of candidates for a question;
- render retrieved text and citations into the generation prompt; and
- measure retrieval quality separately from answer quality and model throughput.

## Prerequisites

- Complete the coding-agent generation interface or provide another function
  that accepts messages and returns generated text.
- Choose a small local corpus with answers that can be checked by hand.
- Define a context budget before indexing. Retrieval must not silently overflow
  the model context or crowd out the user's question.

The Week 3 serving engine is optional for a single-query implementation. Use it
when measuring concurrent RAG requests, not as a prerequisite for learning the
retrieval pipeline.

## Task 1: Build the Corpus

Split each source into chunks that retain a stable source identifier and enough
location metadata to produce a citation. Keep the original text alongside the
embedding input so a retrieved result can be inspected without reverse-engineering
the index.

Checkpoint: reconstructing all chunks for one source preserves their order and
does not lose text at chunk boundaries.

## Task 2: Embed and Index

Compute one embedding per chunk and store it with the chunk identifier. Keep the
first index deliberately simple: an in-memory collection and an explicit
similarity calculation are easier to debug than a service with hidden defaults.

Checkpoint: a query identical to a chunk retrieves that chunk first, and repeated
runs over the same corpus produce the same identifiers.

## Task 3: Retrieve Within a Budget

Retrieve the top candidates, then admit them to the prompt until the evidence
budget is full. Record the score and source metadata for every admitted and
rejected chunk. Do not concatenate an unbounded number of results.

Checkpoint: known-answer queries retrieve at least one supporting chunk, while an
unanswerable query is allowed to return no useful evidence.

## Task 4: Generate a Grounded Answer

Render the question, admitted chunks, and citation identifiers into the prompt.
Ask the model to distinguish statements supported by the supplied evidence from
claims it cannot verify. Return the answer together with the chunk identifiers
used to build the prompt.

Keep this step above the Week 2/3 model boundary. RAG changes input context; it
does not require new attention or quantization kernels.

## Validate the Chapter Checkpoint

Create a small evaluation set containing:

- questions answered by one chunk;
- questions requiring evidence from two chunks;
- distractor chunks with overlapping vocabulary; and
- questions that the corpus cannot answer.

Report retrieval recall or precision, grounded-answer correctness, citation
accuracy, admitted prompt tokens, and time to first token separately. A fluent
answer cannot repair missing evidence, and perfect retrieval can still be lost
through truncation or poor prompt assembly.

For the capstone, retrieve project conventions or API notes and pass the selected
evidence to the coding-agent loop before it chooses a tool.

{{#include copyright.md}}
