"""Optional macOS GPU capture and portable gpudebug reduction for Week 2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROMPT_RULE = "synthetic-token-ids"
PREFILL_LOGITS = "all"
KNOWN_CHECKPOINTS = (
    "kv-cache",
    "quantized-matvec",
    "rmsnorm",
    "rope",
    "swiglu",
    "simd-matmul",
    "context-selected-attention",
    "prefill-fused-gate-up",
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def workload_record(checkpoint: str, phase: str, tokens: int) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint,
        "phase": phase,
        "tokens": tokens,
        "output_tokens": 1,
        "seed": 0,
        "prompt_rule": PROMPT_RULE,
        "prefill_logits": PREFILL_LOGITS,
        "warmup": 1,
        "captured_region": "one-synchronized-forward",
    }


def source_metadata(root: Path = ROOT) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def ensure_outputs_absent(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")


def normalize_capture_outputs(
    trace: Path, metadata: Path, manifest: Path
) -> tuple[Path, Path, Path]:
    outputs = tuple(
        path.expanduser().resolve(strict=False) for path in (trace, metadata, manifest)
    )
    if len(set(outputs)) != len(outputs):
        raise ValueError("capture output paths must be distinct")
    normalized_trace, normalized_metadata, normalized_manifest = outputs
    for output in (normalized_metadata, normalized_manifest):
        if output.is_relative_to(normalized_trace) or normalized_trace.is_relative_to(
            output
        ):
            raise ValueError("trace package must not overlap metadata or manifest")
    return outputs


def package_manifest(package: Path) -> tuple[str, int, int]:
    package = package.resolve(strict=True)
    if not package.is_dir():
        raise ValueError("trace package must be a directory")
    records: list[str] = []
    total_bytes = 0
    for candidate in sorted(package.rglob("*"), key=lambda path: path.as_posix()):
        if candidate.is_symlink():
            raise ValueError(f"trace package contains symlink: {candidate.name}")
        if not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(package)
        except ValueError as exc:
            raise ValueError("trace package member escapes package root") from exc
        if ".." in relative.parts:
            raise ValueError("trace package member contains traversal")
        payload = resolved.read_bytes()
        total_bytes += len(payload)
        records.append(
            f"{hashlib.sha256(payload).hexdigest()}  {relative.as_posix()}\n"
        )
    if not records:
        raise ValueError("trace package contains no files")
    manifest = "".join(records)
    return manifest, total_bytes, len(records)


def verify_manifest(manifest_path: Path, expected_hash: str) -> str:
    payload = manifest_path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_hash:
        raise ValueError(
            f"trace manifest identity mismatch: expected {expected_hash}, got {actual}"
        )
    paths: list[str] = []
    for line in payload.decode().splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("trace manifest contains a malformed record") from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("trace manifest contains an invalid SHA-256")
        member = Path(relative)
        if member.is_absolute() or not relative or ".." in member.parts:
            raise ValueError("trace manifest contains traversal or an absolute path")
        paths.append(member.as_posix())
    if not paths or paths != sorted(set(paths)):
        raise ValueError("trace manifest paths must be unique and sorted")
    return actual


def validate_evidence_identity(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> None:
    fields = (
        ("solution", reference.get("solution"), candidate.get("solution")),
        ("checkpoint", reference.get("checkpoint"), candidate.get("checkpoint")),
        ("phase", reference.get("phase"), candidate.get("phase")),
        (
            "tokens",
            reference.get("workload", {}).get("tokens"),
            candidate.get("workload", {}).get("tokens"),
        ),
        (
            "source tree",
            reference.get("source", {}).get("tree"),
            candidate.get("source", {}).get("tree"),
        ),
        ("workload_id", reference.get("workload_id"), candidate.get("workload_id")),
    )
    for name, expected, actual in fields:
        if expected is None or actual is None:
            raise ValueError(f"evidence identity is missing {name}")
        if expected != actual:
            raise ValueError(f"evidence {name} mismatch")


def iter_json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    cursor = 0
    while cursor < len(text):
        starts = [
            index
            for index in (text.find("{", cursor), text.find("[", cursor))
            if index >= 0
        ]
        if not starts:
            break
        start = min(starts)
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        objects.append(value)
        cursor = end
    return objects


def find_key(value: Any, names: set[str]) -> Any | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower().replace("_", " ") in names:
                return child
        for child in value.values():
            found = find_key(child, names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_key(child, names)
            if found is not None:
                return found
    return None


def availability(value: Any | None, missing_reason: str) -> dict[str, Any]:
    if value is None or value == {} or value == []:
        return {"available": False, "reason": missing_reason}
    return {"available": True, "value": value}


def summarize_gpudebug(objects: list[Any], commands: list[Any]) -> dict[str, Any]:
    if not objects:
        raise ValueError("profile stream contains no usable JSON object")
    timeline = find_key(objects, {"timeline", "gpu timeline"})
    shaders = find_key(objects, {"shaders", "shader ranking", "shader statistics"})
    counters = find_key(objects, {"counters", "counter tree", "gpu counters"})
    counter_fields = {
        name: availability(
            find_key(counters, aliases) if counters is not None else None,
            f"gpudebug emitted no {name} counter",
        )
        for name, aliases in {
            "occupancy": {"occupancy", "simd occupancy"},
            "gpu_duration": {"gpu duration", "duration"},
            "threadgroup_memory": {"threadgroup memory"},
            "memory_bandwidth": {"memory bandwidth", "bandwidth"},
        }.items()
    }
    return {
        "profile_objects": len(objects),
        "timeline": availability(timeline, "gpudebug emitted no timeline tree"),
        "shader_ranking": availability(
            shaders, "gpudebug emitted no shader-ranking tree"
        ),
        "counters": availability(counters, "gpudebug emitted no counter tree"),
        "counter_fields": counter_fields,
        "commands": availability(
            commands if commands else None,
            "no commands stream was supplied or it contained no JSON object",
        ),
    }


def capture(args: argparse.Namespace) -> None:
    raw_outputs = (args.trace, args.metadata, args.manifest)
    ensure_outputs_absent(raw_outputs)
    args.trace, args.metadata, args.manifest = normalize_capture_outputs(*raw_outputs)
    ensure_outputs_absent((args.trace, args.metadata, args.manifest))
    if sys.platform != "darwin":
        raise RuntimeError("capture-week2 requires macOS and Apple Metal")
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("set MTL_CAPTURE_ENABLED=1 before capture-week2")

    import mlx.core as mx
    from mlx_lm import load

    from model_names import shortcut_name_to_full_name

    source_root = ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    models = __import__(f"{args.solution}.models", fromlist=["unused"])
    resolved_model = shortcut_name_to_full_name(args.model)
    mlx_model, _ = load(resolved_model)
    model = models.dispatch_model(
        resolved_model, mlx_model, week=2, checkpoint=args.checkpoint
    )
    inputs = mx.arange(args.tokens, dtype=mx.int32).reshape(1, args.tokens)
    if args.phase == "decode":
        cache = model.create_kv_cache()
        if args.tokens > 1:
            mx.eval(model(inputs[:, :-1], 0, cache))
        capture_inputs = inputs[:, -1:]
        offset = args.tokens - 1
    else:
        cache = model.create_kv_cache()
        capture_inputs = inputs
        offset = 0
    mx.eval(model(capture_inputs, offset, cache, logits_to_keep=None))

    cache = model.create_kv_cache()
    if args.phase == "decode" and args.tokens > 1:
        mx.eval(model(inputs[:, :-1], 0, cache))

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    mx.metal.start_capture(str(args.trace))
    try:
        result = model(capture_inputs, offset, cache, logits_to_keep=None)
        mx.eval(result)
    finally:
        mx.metal.stop_capture()

    manifest, package_bytes, file_count = package_manifest(args.trace)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(manifest)
    manifest_hash = hashlib.sha256(manifest.encode()).hexdigest()
    workload = workload_record(args.checkpoint, args.phase, args.tokens)
    metadata = {
        "schema_version": 1,
        "artifact_kind": "week2-metal-capture",
        "source": source_metadata(),
        "solution": args.solution,
        "model": {"requested": args.model, "resolved": resolved_model},
        "checkpoint": args.checkpoint,
        "phase": args.phase,
        "workload": workload,
        "workload_id": canonical_hash(workload),
        "software": {
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "python": platform.python_version(),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": mx.device_info(),
        },
        "trace_package": {
            "basename": args.trace.name,
            "bytes": package_bytes,
            "files": file_count,
            "manifest_sha256": manifest_hash,
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def reduce(args: argparse.Namespace) -> None:
    ensure_outputs_absent((args.output,))
    metadata = json.loads(args.capture_metadata.read_text())
    required = {
        "source",
        "solution",
        "model",
        "checkpoint",
        "phase",
        "workload",
        "workload_id",
        "trace_package",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"capture metadata is missing {missing[0]}")
    if canonical_hash(metadata["workload"]) != metadata["workload_id"]:
        raise ValueError("capture workload identity mismatch")
    verify_manifest(args.manifest, metadata["trace_package"]["manifest_sha256"])
    profile_objects = iter_json_objects(args.profile_jsonl.read_text())
    command_objects = (
        iter_json_objects(args.commands_jsonl.read_text())
        if args.commands_jsonl is not None
        else []
    )
    result = {
        "schema_version": 1,
        "artifact_kind": "week2-gpudebug-reduction",
        "source": metadata["source"],
        "solution": metadata["solution"],
        "model": metadata["model"],
        "checkpoint": metadata["checkpoint"],
        "phase": metadata["phase"],
        "workload": metadata["workload"],
        "workload_id": metadata["workload_id"],
        "trace_package": metadata["trace_package"],
        "gpudebug": summarize_gpudebug(profile_objects, command_objects),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument(
        "--solution", required=True, choices=("tiny_llm", "tiny_llm_ref")
    )
    capture_parser.add_argument("--model", required=True)
    capture_parser.add_argument(
        "--checkpoint", required=True, choices=KNOWN_CHECKPOINTS
    )
    capture_parser.add_argument("--phase", required=True, choices=("decode", "prefill"))
    capture_parser.add_argument("--tokens", required=True, type=int)
    capture_parser.add_argument("--trace", required=True, type=Path)
    capture_parser.add_argument("--metadata", required=True, type=Path)
    capture_parser.add_argument("--manifest", required=True, type=Path)
    capture_parser.set_defaults(handler=capture)
    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--capture-metadata", required=True, type=Path)
    reduce_parser.add_argument("--manifest", required=True, type=Path)
    reduce_parser.add_argument("--profile-jsonl", required=True, type=Path)
    reduce_parser.add_argument("--commands-jsonl", type=Path)
    reduce_parser.add_argument("--output", required=True, type=Path)
    reduce_parser.set_defaults(handler=reduce)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "tokens") and args.tokens <= 0:
        parser.error("--tokens must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
