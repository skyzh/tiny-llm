# 🚧 Optional: Inspect a Metal Capture

Enter from a completed Week 2 checkpoint and a question left by the portable
attribution result. The example below uses `swiglu`, after the compact primitive
lab. No new learner operator is required: you own the interpretation and choice
of the next experiment. The synchronized product and attribution comparisons
remain sufficient for the required route.

Use this page only if your machine already has the Apple capture tooling needed
by `capture-week2` and `gpudebug`. Check the tools' help for the installed version
before collecting a trace. A missing capture tool is not a failed kernel gate.

## Choose One Region

First rerun the checkpoint's correctness gate and portable attribution:

```bash
pdm run test --week 2 --day 4
pdm run profile-week2-kernels --solution tiny_llm --model qwen3-0.6b \
  --case swiglu:decode:128 --warmup 4 --iterations 12 \
  --json-output week2-capture-control.json
```

Use `rope:decode:128` as the predecessor control if the question concerns the
SwiGLU addition. Keep the same model, phase, and context on both sides.

Create an output directory and choose new filenames; the capture helper refuses
to overwrite its trace, metadata, or manifest:

```bash
mkdir -p out
MTL_CAPTURE_ENABLED=1 pdm run capture-week2 \
  --solution tiny_llm --model qwen3-0.6b \
  --checkpoint swiglu --phase decode --tokens 128 \
  --trace out/swiglu-decode-128.gputrace \
  --metadata out/swiglu-decode-128.capture.json \
  --manifest out/swiglu-decode-128.trace-manifest.sha256
```

The helper warms the chosen shape before capturing a synchronized model region.
Keep the metadata and path-sorted trace manifest with the trace, so a later
inspection can identify the same checkpoint and workload.

## Inspect Only What the Capture Exposes

With `gpudebug` installed, a serialized replay and reduction can use:

```bash
gpudebug --json -t out/swiglu-decode-128.gputrace --timeout 1800 \
  -c 'profile run --gpu-state default --exec serial' \
  > out/swiglu-decode-128.profile.jsonl

pdm run reduce-week2-gpudebug \
  --capture-metadata out/swiglu-decode-128.capture.json \
  --manifest out/swiglu-decode-128.trace-manifest.sha256 \
  --profile-jsonl out/swiglu-decode-128.profile.jsonl \
  --output out/swiglu-decode-128.gpudebug.json
```

If timeline, shader, or counter information is unavailable, leave it unavailable
in your interpretation. A visible dispatch proves that work was encoded; it
does not establish occupancy or a complete-request speedup. A replay's timings
also have their own execution conditions, so return to the matched product run
before deciding whether the operator should stay.

Preserve the compact evidence you need before removing any large trace files.
Continue with the next chapter in the [Week 2 route](./week2-overview.md), or
record the result in your [decision ledger](./week2-decision-ledger.md).

{{#include copyright.md}}
