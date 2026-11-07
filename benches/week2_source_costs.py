"""Optional, version-pinned source-cost view of an existing sampled Metal capture.

Help, input checks and scalar reduction use only the standard library. The private
Apple decoder is loaded only by an explicit read after compatibility checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import sys
from pathlib import Path


FRAMEWORK = Path(
    "/Applications/Xcode.app/Contents/PlugIns/GPUDebugger.ideplugin/Contents/"
    "Frameworks/GTShaderProfiler.framework/Versions/A/GTShaderProfiler"
)
FRAMEWORK_SHA256 = "c6e5ace8bba6fd77c31b5854b782a238b4529247a7feadaa8a1ada269f401b60"
DECODER_SHA256 = "87d1ce330eb19f0d95087ed45ebe7a904eb4d7f56a016b667f1e114fbff92f1f"
TESTED_WITH = "macOS 27.0.1 arm64, Xcode 27.2, Python 3.12.13"


def checked_paths(capture: Path, output: Path) -> tuple[Path, Path, Path]:
    """Refuse an existing/overlapping destination before loading Apple code."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output directory: {output}")
    capture = capture.expanduser().resolve(strict=True)
    output = output.expanduser().resolve(strict=False)
    if not capture.is_dir() or capture.suffix != ".gputrace":
        raise ValueError("capture must be an existing .gputrace directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output directory: {output}")
    if output.is_relative_to(capture) or capture.is_relative_to(output):
        raise ValueError("output must not overlap the input capture")
    streams = list(capture.glob("*.gpuprofiler_raw/streamData"))
    if len(streams) != 1 or not streams[0].is_file():
        raise ValueError("require one embedded sampled profiling session (streamData)")
    if streams[0].stat().st_size >= 4 * 1024**2:
        raise ValueError("streamData exceeds the decoder's 4 MiB input limit")
    return capture, output, streams[0]


def verify_sources(manifest_path: Path) -> dict:
    """Verify optional source identities without exporting their source text."""
    if not manifest_path.exists():
        return {"available": False, "reason": "no source sidecar manifest"}
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("source manifest must be a nonempty file mapping")
    root = manifest_path.parent.resolve(strict=True)
    files = {}
    for source, entry in manifest.items():
        saved = (root / entry["path"]).resolve(strict=True)
        if not saved.is_relative_to(root):
            raise ValueError("source sidecar escapes manifest directory")
        raw = saved.read_bytes()
        if (
            len(raw) != entry["bytes"]
            or hashlib.sha256(raw).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"source sidecar identity mismatch: {source}")
        files[source] = {
            "bytes": len(raw),
            "sha256": entry["sha256"],
            "lines": len(raw.decode("utf-8").splitlines()),
        }
    return {
        "available": True,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "files": files,
    }


def check_compatibility() -> None:
    if not __debug__:
        raise RuntimeError("the pinned decoder requires Python without -O")
    if sys.version_info[:3] != (3, 12, 13):
        raise RuntimeError(
            f"reader needs revalidation with this Python; tested with {TESTED_WITH}"
        )
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(f"optional private reader tested only with {TESTED_WITH}")
    if platform.mac_ver()[0] != "27.0.1":
        raise RuntimeError(
            f"reader needs revalidation on this OS; tested with {TESTED_WITH}"
        )
    if (
        not FRAMEWORK.is_file()
        or hashlib.sha256(FRAMEWORK.read_bytes()).hexdigest() != FRAMEWORK_SHA256
    ):
        raise RuntimeError("GTShaderProfiler differs from the pinned Xcode 27.2 binary")
    decoder = Path(__file__).with_name("_metal_profile_decoder.py")
    if hashlib.sha256(decoder.read_bytes()).hexdigest() != DECODER_SHA256:
        raise RuntimeError("supplied decoder differs from the reviewed source pin")


def decode_existing(stream: Path, output: Path) -> None:
    # The existing decoder may start the installed LLVM helper, but does not
    # create a capture or submit/replay GPU work. Keep its ABI binding isolated.
    reader = importlib.import_module("benches._metal_profile_decoder")
    old_argv, old_cwd = sys.argv, os.getcwd()
    old_run = os.environ.get("FORGE_RUN")
    try:
        os.environ["FORGE_RUN"] = str(output)
        sys.argv = [reader.__file__, str(stream)]
        reader.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        if old_run is None:
            os.environ.pop("FORGE_RUN", None)
        else:
            os.environ["FORGE_RUN"] = old_run


def weighted_cost(row: dict) -> float:
    value = row.get("value0", row.get("alu_cost", 0)) + row.get(
        "value1", row.get("non_alu_cost", 0)
    )
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid weighted instruction cost")
    return value


def cost_rows(data: dict, sources: dict) -> list[dict]:
    """Preserve two attribution modes; inclusive rows are deliberately not summed."""
    rows = []
    for binary in data["binaries"]:
        total = weighted_cost(binary)
        if not math.isclose(
            binary["instruction_cost_sum"], total, abs_tol=1e-7, rel_tol=0
        ):
            raise ValueError("instruction costs do not conserve the binary total")
        outer = sum(weighted_cost(row) for row in binary["leaf_rows"])
        unmapped = binary["unmapped_cost"]
        if not math.isfinite(unmapped) or unmapped < 0:
            raise ValueError("invalid unmapped instruction cost")
        if not math.isclose(outer + unmapped, total, abs_tol=1e-7, rel_tol=0):
            raise ValueError(
                "outer-callsite plus unmapped costs do not conserve the total"
            )
        if not total:
            continue
        for mode, values in (
            ("inclusive", binary["rows"]),
            ("outer-callsite", binary["leaf_rows"]),
        ):
            for row in values:
                value = weighted_cost(row)
                if not value:
                    continue
                source = row.get("file") or ""
                line = row["line"]
                identity = sources.get("files", {}).get(source)
                location = (
                    "unverified-location"
                    if identity is None
                    else (
                        "source-line"
                        if 1 <= line <= identity["lines"]
                        else "compiler-location"
                    )
                )
                rows.append(
                    {
                        "binary_index": binary["index"],
                        "mode": mode,
                        "file": source,
                        "line": line,
                        "function": row.get("function") or "",
                        "location": location,
                        "weighted_cost": value,
                        "percent_of_binary_weighted_cost": 100 * value / total,
                        "instruction_sites": row.get(
                            "num_instructions", row.get("instruction_sites", 0)
                        ),
                    }
                )
    if not rows:
        raise ValueError("capture has no nonzero mapped source costs")
    return sorted(
        rows, key=lambda row: (row["binary_index"], row["mode"], -row["weighted_cost"])
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture",
        type=Path,
        help="existing sampled .gputrace; never captured or replayed here",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="new output directory; must not exist"
    )
    parser.add_argument(
        "--sources-manifest",
        type=Path,
        help="default: sources-manifest.json beside the capture",
    )
    args = parser.parse_args(argv)
    capture, output, stream = checked_paths(args.capture, args.out)
    sources = verify_sources(
        args.sources_manifest or capture.parent / "sources-manifest.json"
    )
    check_compatibility()
    output.mkdir()  # exclusive; parent must already exist, partial failures stay visible
    decode_existing(stream, output)
    data = json.loads((output / "line-cost-data.json").read_text())
    metadata = json.loads((output / "decoded-profile-types.json").read_text())
    rows = cost_rows(data, sources)
    summary = {
        "schema_version": 1,
        "profile_kind": "saved-capture-source-costs",
        "capture": str(capture),
        "tested_with": TESTED_WITH,
        "private_Apple_API": True,
        "framework_sha256": FRAMEWORK_SHA256,
        "decoder_sha256": DECODER_SHA256,
        "new_capture_or_GPU_replay": False,
        "source_text_exported": False,
        "units": "weighted instruction cost, not literal cycles or request-time percentages",
        "inline_limit": "inclusive rows overlap; do not add them; outer-callsite plus unmapped cost is additive",
        "instruction_sites": "static mapped sites, not dynamic execution counts",
        "sources": sources,
        "pipeline_timings": metadata.get("pipeline_timings"),
        "unmapped_costs": [
            {"binary_index": b["index"], "weighted_cost": b["unmapped_cost"]}
            for b in data["binaries"]
        ],
    }
    with (output / "source-costs.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "source-costs.json").open("x") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    with (output / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "source_rows": len(rows),
                "profile_kind": summary["profile_kind"],
            }
        )
    )


if __name__ == "__main__":
    main()
