# Optional: Inspect a Week 2 Capture on macOS 27

The synchronized product benchmark and portable operator-attribution runner
are sufficient for every required Week 2 checkpoint. This page is an optional
deeper look at the same evidence loop for learners with macOS 27 and
`/usr/bin/gpudebug`. It is never an acceptance gate.

The checked example used Qwen3-4B on an Apple M4 Pro at source commit
`add389b747793e910f0506f5720dd0aac373d126`, macOS 27 build `26A428`,
`gpudebug` 1.0, MLX 0.32.0, and mlx-lm 0.31.3. Its product control used a
128-token prompt, 129 output tokens, final-row prefill logits, seed 0, two
warmups, and two balanced fresh-process samples. Its attribution cases used
four warmups and twelve synchronized iterations. These identities bound the
example; they are not a portable timing baseline.

## 1. Prove Correctness First

Choose one checkpoint, phase, and token count. Run its focused test before
capturing it. For a fused-kernel decode example:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k swiglu

pdm run profile-week2-kernels --solution tiny_llm --model qwen3-4b \
  --case swiglu:decode:128 --warmup 4 --iterations 12 \
  --json-output out/swiglu-decode-128.attribution.json
```

The second command is the portable evidence path. Read its checkpoint,
workload, dominant category, and category shares before opening a GPU trace.

## 2. Capture One Synchronized Region

Create `out/` first and choose output names that do not exist. The helper
refuses to overwrite a trace, metadata file, or manifest.

```bash
mkdir -p out

MTL_CAPTURE_ENABLED=1 pdm run capture-week2 \
  --solution tiny_llm --model qwen3-4b \
  --checkpoint swiglu --phase decode --tokens 128 \
  --trace out/swiglu-decode-128.gputrace \
  --metadata out/swiglu-decode-128.capture.json \
  --manifest out/swiglu-decode-128.trace-manifest.sha256
```

The helper compiles and warms the exact shape outside the capture, then
captures one synchronized model region. The metadata records source, model,
checkpoint, phase, token count, prompt rule, software, host, and a canonical
workload identity. The path-sorted manifest hashes every file inside the
`.gputrace` package; treat the package and manifest as one evidence object.

## 3. Replay and Reduce

Run the serialized profile and save the JSON stream. You may also collect
timeline, shader, or command queries into a second JSON-lines file.

```bash
gpudebug --json -t out/swiglu-decode-128.gputrace --timeout 1800 \
  -c 'profile run --gpu-state default --exec serial' \
  > out/swiglu-decode-128.profile.jsonl

pdm run reduce-week2-gpudebug \
  --capture-metadata out/swiglu-decode-128.capture.json \
  --manifest out/swiglu-decode-128.trace-manifest.sha256 \
  --profile-jsonl out/swiglu-decode-128.profile.jsonl \
  --commands-jsonl out/swiglu-decode-128.commands.jsonl \
  --output out/swiglu-decode-128.gpudebug.json
```

If you did not collect command queries, omit `--commands-jsonl`. Missing
timeline, shader, command, or counter trees must remain explicitly unavailable;
do not replace them with zero and do not infer occupancy. In the checked
pre-SIMD 128-token prefill capture, the replay exposed timeline counters but
no shader ranking. In the checked 32-token Split-K capture, only static
dispatch presence was available and no occupancy conclusion was drawn.

## 4. Write a Bounded Decision

Use three sentences:

1. identify the dominant category for this exact checkpoint and workload;
2. name the next bounded change and the same-workload result that would support it;
3. state the result that would falsify the hypothesis or make you revert it.

For example: “At `swiglu:decode:128`, packed projections dominate this M4 Pro
capture and the portable attribution. I will change only the selected
projection schedule and rerun the identical workload. I will revert or choose
another category if projection time does not fall or the complete-model phase
regresses.” This is a reasoning record, not a claim that another device has
the same bottleneck.

## 5. Preserve the Compact Result, Then Clean Up

Keep the capture metadata, manifest, portable attribution, and reduced result
until you have checked their matching workload identity. Raw trace packages
can be enormous; after preserving the compact evidence you need, remove only
the exact trace package and raw streams you created:

```bash
rm -rf -- out/swiglu-decode-128.gputrace
rm -f -- out/swiglu-decode-128.profile.jsonl \
  out/swiglu-decode-128.commands.jsonl
```

The repository includes a compact checked M4 Pro result at
`benchmark_results/m4-pro-qwen3-4b-week2-gpudebug-macos27-mlx-0.32.0.json`.
Learners without macOS 27 can use it to practice reading identity,
availability, dominant categories, and keep/reject decisions. They do not need
to reproduce its exact kernel names, timings, or Metal schedule.

{{#include copyright.md}}
