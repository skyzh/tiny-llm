# Optional saved-capture source costs

The ordinary Week 2 route needs a working learner `capacity-cache` checkpoint,
the model cached during setup, and the supplied request benchmark/operator replay.
Neither that route nor later learner TODOs depend on this private Apple reader.

The supplied `capture-week2` now accepts `--prefill-logits last|all` and
`--cache-capacity C`. It keeps its historical `all` default; pass `last`
explicitly when matching a serving request or the public replay default.
Capacity defaults to the logical context S, with C >= S; `kv-cache` remains
unbounded. Warmup and captured forwards use fresh selected caches, prepare the
decode prefix outside capture, and release every returned cache on success or
failure. Capture metadata schema 2 records those choices. The portable replay
still uses schema 3 and independent operands, so these are different evidence
objects, not interchangeable workload identities.

This additional command reads an **existing sampled** `.gputrace`; it does not
create a capture, run GPU counter replay, or provide a whole-model timeline.

```sh
python -m benches.week2_source_costs --help

# In the prepared course environment, after obtaining a sampled capture:
pdm run source-costs-week2 path/to/saved.gputrace --out out/new-source-costs
```

Create the output's parent directory first. The destination itself must not
exist, including an empty directory or dangling symlink. Keep the capture's
`.gpuprofiler_raw` data and metallib/metallibsym companions together. A capture
without one embedded sampled `streamData` session is refused; merely saving an
ordinary capture does not guarantee those samples exist.

Keep `sources-manifest.json` and `sources/` beside the capture, or pass
`--sources-manifest PATH`, to verify source identities. Sidecar contents are
hash/length checked but never copied into the outputs. Without the manifest,
locations remain explicitly unverified. Compiler locations outside the verified
source's physical line range are labelled separately. Apple SDK sources and raw
captures are not included in this repository.

## Compatibility and provenance

The underlying decoder is the actual v2 `read_mio_profile.py`, supplied as
`_metal_profile_decoder.py` without source changes. Its SHA256 is
`87d1ce330eb19f0d95087ed45ebe7a904eb4d7f56a016b667f1e114fbff92f1f`.
The original `shader_source_costs.py` wrapper has SHA256
`5411af901eaa9174e237bde9bc7ecb31f51cb9a65acb4ed01639a87db51f097a`;
the course wrapper replaces source-text/HTML export with code-free CSV/JSON.

That original decoder was exercised on macOS **27.0.1 arm64**, Xcode **27.2**,
Python **3.12.13**. This entrypoint refuses another OS/Python version and a
GTShaderProfiler binary different from the pinned Xcode build. It expects Xcode
at `/Applications/Xcode.app`, including its installed GTLLVMHelper. Private APIs
and helper behavior need revalidation after a toolchain update; a matching hash
is not a general compatibility or zero-system-I/O guarantee. Do not use Python
`-O`, which disables the decoder's ABI/conservation assertions.

Only an explicit read loads GTShaderProfiler/Foundation/Objective-C via ctypes
and may start the installed LLVM helper. `--help` needs only the standard
library. No installation or shared environment change is performed by the tool;
the existing PDM `pre_run` hook may synchronize the prepared course environment.
The direct `python -m` form avoids that PDM hook. The course packaging has not
yet been exercised against Apple frameworks; the original v2 result is prior
evidence, not validation of this new entrypoint.

## Read the outputs

- `source-costs.csv` / `source-costs.json`: binary index, attribution mode,
  file/line/function, source-identity status, weighted cost, percent of that
  binary's weighted cost, and static instruction-site count. No source text.
- `summary.json`: compatibility/source hashes, unavailable-source status,
  unmapped costs, limitations, and the decoder's separate pipeline timings.
- `line-cost-data.json` / `decoded-profile-types.json`: the decoder's scalar
  metadata and source locations. These contain no tensor/buffer or source-body
  exports. Partial failures remain in the new directory for diagnosis.

Weighted instruction cost is **not literal cycles or a request-time percentage**.
Inclusive inline rows overlap and must not be added. Outer-callsite rows plus
unmapped cost conserve each binary total, but optimized/inlined code can collapse
onto a caller. Static sites are not dynamic execution counts. Keep pipeline
timings, operator replay, and ordinary synchronized request timing separate.

Use a hot location to form a hypothesis, then test the mechanism and remeasure
the complete matched request. No source-cost ranking by itself establishes a
hardware bottleneck, model wiring, numerical correctness, or an optimization win.
