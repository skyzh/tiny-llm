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

## Use Xcode's Metal Debugger on macOS 26

On macOS 26, use the graphical Metal debugger that ships with Xcode. The
`gpudebug` executable is an operating-system tool installed at
`/usr/bin/gpudebug` by macOS 27; installing Xcode 27 beta, Command Line Tools,
or a Metal Toolchain does not add that executable to macOS 26. Do not install
or select a beta Xcode merely to look for it.

Select the Xcode installation that will compile, capture, and replay the
course kernels, then verify that all four commands resolve through it:

```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

sw_vers -productVersion
xcode-select --print-path
xcodebuild -version
xcrun metal --version
```

The reference environment uses macOS 26.5.2, Xcode 26.6, and the Metal 17.6
compiler selected by that Xcode. Rebuild the extension after changing Xcode or
Metal toolchains; otherwise a new trace can still contain a metallib produced
by the old compiler.

On macOS 27, Apple's
[AI-agent workflow](https://developer.apple.com/documentation/xcode/investigating-gpu-issues-with-ai-agents)
can inspect the same trace through `/usr/bin/gpudebug`. That interface is
optional. The workflow remains compatible with macOS 26 by driving Xcode's
graphical debugger through Computer Use and preserving whole-window screenshots
from a large normal window. You can perform the same clicks manually in Xcode;
the automation changes how the evidence is collected, not what Xcode measures.

## Match the Tool to the Question

| Question | Measurement |
|---|---|
| Did the complete model improve? | Fresh-process throughput benchmark |
| Which operator family dominates? | Synchronized kernel-group attribution |
| Which shader, function, and source line is expensive? | Xcode Performance views and Shader Cost Graph |

The operator-attribution chart is not a flame graph. On M3 and newer Macs,
the Metal debugger's
[Shader Cost Graph](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-shader-cost-graph-a17-m3)
ranks shader function calls and connects them to weighted source lines.
[Pipeline Statistics](https://developer.apple.com/documentation/xcode/analyzing-draw-command-and-compute-dispatch-performance-with-pipeline-statistics)
provides instruction, ALU, cache, MMU, control-flow, register, and spill
evidence for a selected pipeline.

## Capture One Course-Owned Shader

Build your extension with source and line tables, then capture one Qwen3-4B
checkpoint at its real shape:

```bash
CMAKE_ARGS="-DMLX_METAL_DEBUG=ON" pdm run build-ext

MLX_METAL_DEBUG=1 MTL_CAPTURE_ENABLED=1 pdm run capture-week2-shader \
  --solution tiny_llm \
  --workload quantized-projection \
  --projection q --rows 1 \
  --schedule matvec \
  --iterations 10 \
  --output /tmp/week2-q-projection.gputrace
```

The capture uses synthetic buffers with the real
`A[M,N] @ W[K,N]^T`, `M=1`, `N=2560`, `K=4096` Qwen3-4B shape. The
dispatched kernel and schedule are unchanged, while model weights do not need
to be embedded in the trace. Warmup and input materialization happen before
capture.

Use `--solution tiny_llm_ref` with `build-ext-ref` only when reproducing the
reference evidence ledger. Do not profile the reference solution and treat its
bottleneck as proof about your implementation.

The capture helper also accepts `--workload dense-projection`,
`--workload pointwise`, and `--workload decode-attention`. Use
`--schedule vanilla` for the vanilla Metal quantized projection control. The
matrix-shaped controls use `--rows 32 --schedule simd-matmul` or `split-k`.

Do not validate a trace by file size. After replay, require all three checks:

- the exact target pipeline appears;
- Xcode's Summary reports at least one compute encoder and dispatch;
- profiling produces nonzero GPU time and counter rows.

`--iterations` is the requested evaluation count, not a promised dispatch
count. MLX may materialize or synchronize the graph differently, so record
the replay summary.

## Replay and Profile in Xcode

Open the trace in the same Xcode installation that selected the Metal
compiler:

```bash
open -a /Applications/Xcode.app /tmp/week2-q-projection.gputrace
```

In the trace window:

1. Select **Profile after replay**, then choose **Replay**.
2. Open **Performance** in a large normal window. Use **Overview** for effective
   GPU time, encoder, pipeline-state, and GPU-command counts, performance state,
   and the Top Shaders table with names, cost, SIMD groups, register allocation,
   high-water mark, and spills. Then use **Shaders** to select the dominant
   pipeline before opening its counters.
3. In **Counters**, capture **Performance Limiters** twice when necessary: the
   left columns for occupancy, instruction throughput, ALU, and F32 evidence;
   the right columns for MMU and last-level-cache evidence.
4. Capture **Memory** twice when necessary: bandwidth and read/write rates on
   the left, then transferred bytes, cache traffic, and miss rates on the
   right. Captured resource size is not a substitute for bytes read by one
   dispatch.
5. Open **Cost Graph**, select the dominant pipeline, and drag the source pane
   upward until it occupies about two-thirds of the window. Show 20–30 lines
   around the hottest loop with the per-line weighted percentages, plus the
   instruction and data-type cost summaries. Resize the normal window or source
   pane until all three are legible in the same image.

Xcode's replay duration is a diagnostic measurement rather than an end-to-end
throughput result. Record the execution mode shown in **Overview** instead of
assuming serialization. Keep the trace window open until every evidence view
is captured; reopening and reprofiling a large trace can take minutes.

## Preserve the Same Screenshot Set

Save the same six images for every Day 2–7 reference checkpoint:

```text
week2-dayN-xcode-overview.png
week2-dayN-xcode-limiters-left.png
week2-dayN-xcode-limiters-right.png
week2-dayN-xcode-memory-left.png
week2-dayN-xcode-memory-right.png
week2-dayN-xcode-cost-source.png
```

The overview must keep every relevant pipeline visible. Day 4 therefore shows
RMSNorm, RoPE, and SwiGLU together; Day 7 shows both the Split-K accumulation
and reduction when the reduction has material cost. The Cost Graph image is
the critical source attachment: a function name without the hottest loop and
its weighted source-line percentages is incomplete.

Place the trace name, source commit, implementation, tensor shape, hardware,
macOS, Xcode, Metal compiler, MLX version, and performance state in the figure
caption or the surrounding checkpoint prose. These provenance fields apply to
the screenshot set; they do not need to be overlaid on every image.

Counter and source-cost percentages are comparable within one replay. They are
not percentages of end-to-end model time, and the measured values are not
targets for another machine. The
[M4 Pro evidence ledger](./appendix-performance.md#week-2-xcode-checkpoint-contract)
defines where each screenshot set belongs beside the operator and model
measurements that give it meaning.

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
