# 🚧 Week 2 Advanced Appendix: Metal Profiling

> **Status: Optional, hardware-specific investigation.** This appendix is not
> required to complete the core Week 2 course. See the
> [verification matrix](./week2-overview.md#verification-status) before
> generalizing a result from one GPU.

Use this appendix after a synchronized operator benchmark and the
dependency-aware kernel attribution agree on the kernel family to investigate.
The required Day 2 lab ends with benchmark JSON and an attribution profile;
GPU trace replay, counter interpretation, source-line analysis, and schedule
tuning belong here.

## Configure Agent-Compatible GPU Tools on macOS 26

`gpudebug` is documented as part of the Xcode 27 command-line workflow, but
Xcode 27 beta does not require macOS 27. Apple supports the beta on macOS 26.4
or later. Install Xcode beta from
[Apple Developer Downloads](https://developer.apple.com/download/all/), then
select it and install its optional Metal compiler toolchain:

```bash
sudo xcode-select --switch /Applications/Xcode-beta.app/Contents/Developer
xcodebuild -runFirstLaunch -checkForNewerComponents
xcodebuild -downloadComponent MetalToolchain
```

The Xcode app, standalone Command Line Tools package, and Metal Toolchain are
distinct pieces. Selecting the Xcode beta does not install the optional Metal
compiler, and installing that compiler does not replace an older standalone
Command Line Tools package. The standalone package is not a substitute for
selecting the Xcode app. Verify the active versions and the actual executable
instead of assuming that any one installer provided everything:

```bash
sw_vers -productVersion
xcode-select --print-path
xcodebuild -version
pkgutil --pkg-info=com.apple.pkg.CLTools_Executables
xcrun metal --version
command -v gpudebug || xcrun --find gpudebug
man gpudebug
```

If `metal` is missing, repeat the `xcodebuild -downloadComponent` command. Do
not switch the active developer path back to
`/Library/Developer/CommandLineTools`; the build and capture commands below
need the selected Xcode beta.

> **Xcode 27 beta 4 packaging note:** On macOS 26.5.2, Xcode 27 beta 4
> (`27A5228h`), its `27A5228f` Metal Toolchain, and Command Line Tools 27 beta 4
> install the `gpudebug(1)` manual page but not a discoverable `gpudebug`
> executable. A successful `man gpudebug` is therefore not an installation
> check. Require `command -v` or `xcrun --find`; when both fail, use the Xcode 26
> GUI fallback below for the trace and retain the text evidence format. Do not
> infer that macOS 27 is required from this beta packaging gap.

Xcode 26's graphical Metal debugger can open the same `.gputrace` and expose
the counter tables, shader list, and Shader Cost Graph used in this appendix.
That is a valid manual fallback. The course uses the Xcode beta `gpudebug`
interface when available because its self-describing text and JSON output can
be inspected, recorded, and compared by a coding agent without screenshots.
See Apple's
[AI-agent workflow](https://developer.apple.com/documentation/xcode/investigating-gpu-issues-with-ai-agents)
and run `man gpudebug` for the installed command reference.

## Match the Tool to the Question

| Question | Measurement |
|---|---|
| Did the complete model improve? | Fresh-process throughput benchmark |
| Which operator family dominates? | Synchronized kernel-group attribution |
| Which shader, function, and source line is expensive? | `gpudebug` performance tree and source costs |

The operator-attribution chart is not a flame graph. On M3 and newer Macs,
the Metal debugger's
[Shader Cost Graph](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-shader-cost-graph-a17-m3)
ranks shader function calls and connects them to weighted source lines.
[Pipeline Statistics](https://developer.apple.com/documentation/xcode/analyzing-draw-command-and-compute-dispatch-performance-with-pipeline-statistics)
provides instruction, ALU, cache, MMU, control-flow, register, and spill
evidence for a selected pipeline.

## Capture One Course-Owned Shader

Build your extension with source and line tables, then capture one Qwen3-4B
projection at its real shape:

```bash
CMAKE_ARGS="-DMLX_METAL_DEBUG=ON" pdm run build-ext

MTL_CAPTURE_ENABLED=1 pdm run capture-week2-shader \
  --solution tiny_llm \
  --projection q --rows 1 \
  --iterations 10 \
  --output /tmp/week2-q-projection.gputrace
```

The capture uses synthetic buffers with the real `M=1`, `K=2560`, `N=4096`
Qwen3-4B shape. The dispatched kernel and schedule are unchanged, while model
weights do not need to be embedded in the trace. Warmup and input
materialization happen before capture.

Use `--solution tiny_llm_ref` with `build-ext-ref` only when reproducing the
reference evidence ledger. Do not profile the reference solution and treat its
bottleneck as proof about your implementation.

Do not validate a trace by file size. After replay, require all three checks:

- the exact target pipeline appears;
- `gpudebug` reports at least one compute encoder and dispatch;
- profiling produces nonzero GPU time and counter rows.

`--iterations` is the requested evaluation count, not a promised dispatch
count. MLX may materialize or synchronize the graph differently, so record
the replay summary.

## Reuse One `gpudebug` Session

Point an agent at the trace and `man gpudebug`. Begin with a single session and
reuse its printed session identifier so that the trace and replayer are not
loaded for every question:

```bash
gpudebug -t /tmp/week2-q-projection.gputrace -c "list"

# Replace 412 with the session identifier printed above.
gpudebug -s 412 -c "profile list"

# For a raw capture, collect and embed a new profile.
gpudebug -s 412 \
  -c "profile run --gpu-state medium --exec serial --embed"
gpudebug -s 412 -c "go performance" -c "list --all"
```

If `profile list` reports an embedded session that you intend to reuse, run
`profile load` instead of `profile run`. Do not load one profile and then
silently replace it with another.

The available performance nodes depend on the GPU and trace. Follow the
actions printed by `list`; do not assume a hard-coded path. Use `go`, `find`,
`info`, and `list --all` to locate the target pipeline, its encoders,
shader statistics, counters, and weighted source lines. Add `--json` when a
result needs to be aggregated or compared programmatically. End the session
when the evidence record is complete:

```bash
gpudebug --terminate 412
```

Use `--oneshot` only for an isolated query. A chapter investigation normally
needs several commands, so repeatedly loading the trace with `--oneshot` wastes
time and can make the investigation harder to follow.

## Record Text Evidence, Not Screenshots

Save a compact text or JSON record beside the benchmark result. It must contain:

- trace path, source commit, hardware, macOS, Xcode, MLX, and performance state;
- tensor shape, exact pipeline, compute-encoder count, and dispatch count;
- total GPU time and steady-state median dispatch time;
- allocated registers, register high-water mark, and spilled bytes;
- occupancy, instruction, arithmetic, cache, MMU, and control-flow limiters;
- bytes read per dispatch, memory bandwidth, cache bandwidth, and miss rate;
- highest-cost source lines as filename, line number, code, and cost percentage.

Treat the first recorded dispatch as replay warmup and report the median of the
remaining rows. Use the same exclusion rule for every comparison. Bandwidth
describes transfer rate; bytes per dispatch describes how much traffic the
algorithm requires. Keep both.

The reference decode-matvec trace, for example, attributes 71.85% of shader
cost to four adjacent expressions:

| Line | Metal source | Shader cost |
|---:|---|---:|
| 516 | `scaled_activations[local] * (weights & 0x000f)` | 22.44% |
| 517 | `scaled_activations[local + 1] * (weights & 0x00f0)` | 20.38% |
| 518 | `scaled_activations[local + 2] * (weights & 0x0f00)` | 16.20% |
| 519 | `scaled_activations[local + 3] * (weights & 0xf000)` | 12.83% |

Counter and source-cost percentages are comparable within one replay. They are
not percentages of end-to-end model time, and the measured values are not
targets for another machine. The
[M4 Pro evidence ledger](./appendix-performance.md#m4-pro-decode-matvec-pipeline-profile)
shows how raw rows become median tables and a bounded interpretation.

Missing source lines mean the extension was not rebuilt with
`MLX_METAL_DEBUG`. Missing counter samples mean the profiler is unsupported on
that OS, Xcode, or GPU combination. Neither result justifies an ALU- or
bandwidth-bound claim.

## Longer Traces with Instruments

For a longer request, Instruments can complement the single-dispatch capture:

```bash
xcrun xctrace list templates

xcrun xctrace record \
  --template /path/to/TinyLLMMetal.tracetemplate \
  --output /tmp/week2.trace \
  --launch -- pdm run bench-week2-operators \
    --solution tiny_llm --model qwen3-4b \
    --section prefill-projections --context 32 --prefill-projection k

xcrun xctrace export --input /tmp/week2.trace --toc
```

The stock Metal System Trace is useful for command buffers, queues, and GPU
intervals. A compatible Metal Shader Timeline or counter template can rank
pipelines over a longer request.

Do not use trace-instrumented wall time as a throughput result: capture adds
overhead and may serialize commands. Preserve the same evidence fields as the
single-shader trace so the longer trace answers a specific new question.

## Evidence Order

Use the tools in this order:

1. Save the acceptance benchmark JSON outside a trace.
2. Use operator attribution to select a kernel family.
3. Capture that kernel and inspect its pipeline, function, and source-line
   costs.
4. Change one schedule or operation justified by the evidence.
5. Re-run the isolated operator, end-to-end benchmark, and attribution profile.

If a trace does not identify a dominant cost, do not invent one from the source
code. Shorten the workload or return to dependency-aware attribution. Retain an
experiment only when its isolated result and complete-model phase move in the
same direction.

{{#include copyright.md}}
