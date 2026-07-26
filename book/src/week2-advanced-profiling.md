# 🚧 Week 2 Advanced Appendix: Metal Profiling

> **Status: Optional, hardware-specific investigation.** This appendix is not
> required to complete the core Week 2 course. See the
> [verification matrix](./week2-overview.md#verification-status) before
> generalizing a result from one GPU.

Use this appendix after a synchronized operator benchmark and the
dependency-aware kernel attribution agree on the kernel family to investigate.
The required Day 2 lab ends with benchmark JSON and an attribution profile;
Xcode capture, counter interpretation, source-line analysis, and schedule
tuning belong here.

## Match the Tool to the Question

| Question | Measurement |
|---|---|
| Did the complete model improve? | Fresh-process throughput benchmark |
| Which operator family dominates? | Synchronized kernel-group attribution |
| Which shader, function, and source line is expensive? | Metal Pipeline Statistics and Shader Cost Graph |

The operator-attribution chart is not a flame graph. On M3 and newer Macs,
Xcode's
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

open /tmp/week2-q-projection.gputrace
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
- Xcode reports at least one compute encoder and dispatch;
- profiling produces nonzero GPU time and counter rows.

`--iterations` is the requested evaluation count, not a promised dispatch
count. MLX may materialize or synchronize the graph differently, so record
Xcode's replay summary.

## Capture the Counter Tables

Open the trace in Xcode, click the profiling gauge, and wait for replay:

1. Open **Counters**, select **Encoders**, and filter to the repeated target
   compute encoders.
2. Select **Performance Limiters**. Expose occupancy, instruction throughput,
   integer and complex, F32, ALU, MMU, last-level-cache, and control-flow
   columns.
3. Confirm that every row belongs to the target pipeline.
4. Treat the first recorded dispatch as replay warmup and report the median of
   the remaining rows. Use the same exclusion rule for every comparison.

![Xcode Performance Limiters table for repeated decode-matvec dispatches](./week2-xcode-arithmetic-counters.png)

Switch to **Memory** without changing the encoder selection. Record
device-memory bandwidth, GPU read bandwidth, bytes read from device memory,
last-level-cache bandwidth, and cache miss rate.

![Xcode Memory table for the same repeated dispatches](./week2-xcode-bandwidth-counters.png)

Bandwidth and bytes answer different questions: bandwidth describes transfer
rate, while bytes per dispatch describes how much traffic the algorithm
requires. Preserve the column headers and several dispatch rows in screenshots,
and record the raw values separately. A crop of unexplained numbers is not
reproducible evidence.

## Capture the Shader Cost Graph

The limiter table selects a kind of work. The Shader Cost Graph locates that
work in the program:

1. Open **Shaders** and find the target pipeline. Record GPU time, allocated
   registers, register high-water mark, and spilled bytes.
2. Double-click the pipeline-state cell, then open **Cost Graph**.
3. Follow the highest-cost function node. In **Source Files**, select the Metal
   source and keep the source metric set to **Cost**.
4. Record the highest-cost lines and percentages with the pipeline name,
   source filename, line numbers, and cost labels visible.

![Xcode Shader Cost Graph for the masked W4 dot product](./week2-xcode-matvec-hot-lines.png)

Counter and source-cost percentages are comparable within one replay. They are
not percentages of end-to-end model time, and the screenshots are examples of
the workflow rather than targets for another machine. The
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
overhead and may serialize commands. Record at least the tensor shape, pipeline
name, GPU time, dispatch count, Pipeline Statistics activity, highest-cost
function, and highest-cost source line.

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
