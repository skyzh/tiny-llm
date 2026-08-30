# Day 9: Bound Tool Evidence

An agent can read a log that is much larger than the useful part. Appending the
whole result to every later model prompt wastes context, but silently slicing
it loses the evidence needed to verify what happened.

Day 9 keeps those two concerns separate:

- preserve the exact UTF-8 tool-result bytes outside the prompt;
- give the model a bounded observation with identity, size, digest, and
  head/tail previews;
- let the model request one explicit byte range and continue in the unchanged
  agent loop.

This is byte selection, not semantic summarization. The model decides which
range to inspect from visible facts.

## Files You Implement

This is the final declaration scaffold, so every earlier Week 4 module is
already visible. Day 9 owns only the TODO bodies in the evidence adapter and
its exports:

| File | Public names | Responsibility |
| --- | --- | --- |
| `src/tiny_llm/agent/evidence.py` | `ArtifactRef`, `ArtifactStore`, `BoundedEvidenceWorkspace` | Store exact results, render bounded observations, and serve explicit ranges. |
| `src/tiny_llm/agent/__init__.py` | the names above | Export the completed Day 9 declaration surface. |

The protocol, loop, workspace, receipts, and Days 1–8 modules do not change.
`BoundedEvidenceWorkspace` is a small adapter around the existing `Workspace`.

Copy the Day 9 test into the learner workspace:

```bash
pdm run copy-test --week 4 --day 9
pdm run test --week 4 --day 9
```

Before you implement the TODOs, the implementation-dependent cases across six
tasks are expected to fail; the shared constructor-validation cases already pass.
The test is day-local and does not rerun Days 1--8. `copy-test` refreshes the
learner copy from the supplied checkpoint.

Use this command for the supplied implementation:

```bash
pdm run test-refsol --week 4 --day 9
```

## Task 1: Give Exact Bytes an Identity

`ArtifactStore.put(result)` encodes the complete result as UTF-8, writes those
bytes under its explicit artifact root, and returns:

```python
ArtifactRef(
    artifact_id="artifact-<lowercase SHA-256>",
    byte_count=...,
    sha256="<lowercase SHA-256>",
)
```

The content-addressed ID and full digest deliberately repeat the same hash in
different roles: one is the handle used by the range request; the other is a
separately labeled model-visible verification field. The store registers the
ID in memory. A different store cannot retrieve it merely because the caller
guessed the filename.

Before every range read, verify the stored byte count and digest again. The
course store is local and single-process. It does not promise retention,
garbage collection, encryption, access control, or a network blob service. It
preserves the exact bytes returned by the wrapped tool; earlier tool-level
limits, such as Day 3's command-output cap, still apply before this adapter.

## Task 2: Replace Only Oversized Successful Results

Wrap an existing workspace:

```python
from tiny_llm.agent import ArtifactStore, BoundedEvidenceWorkspace

bounded = BoundedEvidenceWorkspace(
    workspace,
    ArtifactStore(artifact_root),
    max_inline_bytes=512,
    preview_bytes=64,
    max_range_bytes=512,
)
```

Short results and every `error:` observation remain byte-for-byte unchanged.
For a successful result larger than `max_inline_bytes`, persist the full bytes
and return a compact JSON observation containing:

- `artifact_id`, `byte_count`, and `sha256`;
- valid UTF-8 head and tail previews with their byte ranges;
- the omitted half-open byte interval;
- one exact `read_file` range-request example.

The entire compact observation, including metadata and previews, must fit
`max_inline_bytes`. Reduce previews at UTF-8 boundaries when the metadata needs
more space. The supplied constructor already requires
`max_range_bytes >= 4`, so the default range can always hold one maximum-width
UTF-8 code point. Preserve that supplied rule; the remaining externalization
and range behavior is learner-owned. Never split a code point or silently
replace one.

## Task 3: Reuse the Existing Tool Protocol

Day 9 does not add a new action schema. It reserves one virtual relative-path
namespace for the existing `read_file` action:

```text
.tool-artifacts/<artifact-id>/bytes/<start>-<end>
```

`[start,end)` is an exact half-open byte range. The adapter intercepts the
reserved prefix before the real workspace sees it. A successful reply names
the same artifact, total size, digest, start, end, returned byte count, and the
strictly decoded UTF-8 data.

The reply is not sent back through externalization. Its selected data is
already limited by `max_range_bytes`.

## Task 4: Fail Closed Without Leaking

Every path beginning with `.tool-artifacts/` belongs to the virtual namespace.
Malformed paths must not fall through to a learner file of the same name.

Return short ordinary `error:` observations for:

- an invalid or unknown artifact ID;
- negative, reversed, out-of-bounds, or oversized ranges;
- stored bytes whose size or digest changed;
- a range that cuts through a UTF-8 code point.

Do not print the host artifact-root path, enumerate known IDs, or reveal bytes
from another store while reporting an error.

## Task 5: Continue Through the Same Loop

The deterministic test creates a large ASCII build log whose diagnostic is
outside both previews. A scripted model performs three normal steps:

```text
read_file build.log
        |
        v
bounded identity + previews
        |
        v
read_file .tool-artifacts/<id>/bytes/<start>-<end>
        |
        v
exact diagnostic range -> final answer
```

`run_agent` is unchanged. Its first event contains only the bounded
observation; the second contains only the selected range; the artifact file
still matches the complete original result.

Before running it, predict whether the first result stays inline or becomes an
artifact, the omitted half-open byte interval, and the exact bytes returned by
the model's range request. The first observation, artifact digest/file, and
second observation must agree; a final model sentence is not the evidence.

## Task 6: Preserve the Workspace Contract

Delegate `policy`, `available_tools`, and `modified_files` to the wrapped
workspace. This lets `build_system_prompt`, action validation, and the existing
event loop operate without knowing about the storage adapter.

The virtual range path is still a normal JSON `read_file` request, so the
learner does not need a second parser or a replacement generation interface.

## Manual Cached-Qwen Walkthrough

Complete the Day 9 TODOs first. Create separate disposable workspace and
artifact directories, put a large UTF-8 `build.log` in the workspace, and use
the same local-model adapter as the exploratory Week 4 exercise:

```python
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_lm import generate as mlx_generate, load
from tiny_llm.agent import (
    ArtifactStore,
    BoundedEvidenceWorkspace,
    ToolPolicy,
    Workspace,
    run_agent,
)

workspace_directory = TemporaryDirectory(prefix="tiny-llm-day9-workspace-")
artifact_directory = TemporaryDirectory(prefix="tiny-llm-day9-artifacts-")
workspace_root = Path(workspace_directory.name)
artifact_root = Path(artifact_directory.name)
(workspace_root / "build.log").write_text(
    "build started α\n"
    + "x" * 256
    + "\nERROR code=E42 dependency mismatch\n"
    + "y" * 3_000,
    encoding="utf-8",
)

mlx_model, tokenizer = load("Qwen/Qwen3-0.6B-MLX-4bit")
artifacts = ArtifactStore(artifact_root)
workspace = BoundedEvidenceWorkspace(
    Workspace(ToolPolicy(workspace_root, max_file_bytes=64_000)),
    artifacts,
)

def generate(messages):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return mlx_generate(
        mlx_model, tokenizer, prompt, max_tokens=256, verbose=False
    )

run = run_agent(
    "Read build.log. If it is externalized, retrieve one useful byte range.",
    generate,
    workspace,
)

for event in run.events:
    print(event.result)
print(run.final)
for artifact_path in artifact_root.iterdir():
    data = artifact_path.read_bytes()
    print(artifact_path.name, len(data), hashlib.sha256(data).hexdigest())
```

Save that fragment as `/tmp/tiny-llm-day9.py`, then run the repository source
explicitly so an older editable install cannot be imported by accident:

```bash
PYTHONPATH=src pdm run python /tmp/tiny-llm-day9.py
```

Model choices vary. Inspect the actual first observation, requested artifact
ID and range, returned bytes, final answer, and on-disk artifact digest. Do not
use a workspace or artifact root containing secrets. After inspection, call
`workspace_directory.cleanup()` and `artifact_directory.cleanup()`.

Invalid or repeated model JSON is an ordinary bounded outcome: the loop may
stop at its invalid-action or step budget without externalizing anything. That
does not invalidate the deterministic checkpoint, and a successful manual run
does not become a correctness or performance result.

## Checkpoint

You can now keep a complete large tool result available for verification while
placing only bounded facts in the model context. The model can retrieve an
explicit range by identity and continue through the same tokenizer and agent
loop.

Day 9 does not summarize the result, stream concurrent chunks, retain artifacts
for production, or add a network service.

The current checkpoint starts a standalone bounded-evidence loop. It does not
yet receive Day 5's compacted transcript or Day 8's selected branch. The
pending cumulative capstone will supply that composed handoff and report token,
reuse, evaluation, and byte counts together. Those counts remain accounting
evidence; they do not prove latency, throughput, quality, or memory-capacity
improvement.

{{#include copyright.md}}
