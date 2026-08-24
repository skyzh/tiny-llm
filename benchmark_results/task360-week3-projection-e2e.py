"""Balanced native E2E evidence for the Week 3 MLX projection seam.

Run from the repository root with the project environment active:

    HF_HUB_OFFLINE=1 python benchmark_results/task360-week3-projection-e2e.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "benchmark_results"
MODEL = "qwen3-0.6b"


def run(label: str, arguments: list[str]) -> dict:
    output = OUTPUT_DIR / f"task360-{label}.json"
    command = [sys.executable, "-m", *arguments, "--json-output", str(output)]
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    source_path = str(ROOT / "src")
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not current_pythonpath
        else source_path + os.pathsep + current_pythonpath
    )
    print(f"[{label}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return json.loads(output.read_text())


def trace_hash(payload: dict) -> str | None:
    trace = payload.get("request_trace")
    if trace is None:
        return None
    encoded = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_metadata() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status:
        raise RuntimeError("benchmark source must be tracked-clean")
    return {"git_commit": commit, "git_tree": tree, "git_tracked_dirty": False}


def main() -> None:
    source = source_metadata()
    manifest = {
        "week2": [],
        "chunked": [],
        "day3": [],
        "serving": [],
    }

    week2 = run(
        "week2-course",
        [
            "benches.bench_course_progression",
            "--suite",
            "week2",
            "--solution",
            "ref",
            "--model",
            MODEL,
            "--variant",
            "week2-simd-matmul",
            "--variant",
            "week2-split-k",
            "--variant",
            "mlx",
            "--input-len",
            "128",
            "--output-len",
            "33",
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--prefill-logits",
            "last",
            "--offline",
            "--cooldown-seconds",
            "0.5",
        ],
    )
    manifest["week2"].append(
        {
            "label": "course",
            "source": week2["source"],
        }
    )

    for step in (512, 128):
        for sequence, mode in enumerate(
            ("seam", "inherited", "inherited", "seam"),
            start=1,
        ):
            extra = [] if mode == "seam" else ["--week3-inherit-course-projections"]
            label = f"chunked-{step}-{sequence:02d}-{mode}"
            payload = run(
                label,
                [
                    "benches.bench",
                    "--solution",
                    "ref",
                    "--loader",
                    "week2",
                    "--model",
                    MODEL,
                    "--batch-decode",
                    "--num-seqs",
                    "8",
                    "--batch-size",
                    "4",
                    "--min-input-len",
                    "64",
                    "--max-input-len",
                    "512",
                    "--min-output-len",
                    "32",
                    "--max-output-len",
                    "32",
                    "--prefill-step",
                    str(step),
                    "--prefill-logits",
                    "last",
                    "--warmup",
                    "1",
                    "--seed",
                    "0",
                    *extra,
                ],
            )
            manifest["chunked"].append(
                {
                    "label": label,
                    "mode": mode,
                    "prefill_step": step,
                    "source": payload.get("source", source),
                    "trace_sha256": trace_hash(payload),
                }
            )

    for sequence, mode in enumerate(
        ("seam", "inherited", "mlx", "mlx", "inherited", "seam"),
        start=1,
    ):
        if mode == "mlx":
            solution = "mlx"
            extra = []
        else:
            solution = "ref"
            extra = [] if mode == "seam" else ["--week3-inherit-course-projections"]
        label = f"day3-{sequence:02d}-{mode}"
        payload = run(
            label,
            [
                "benches.bench",
                "--solution",
                solution,
                "--loader",
                "week3",
                "--model",
                MODEL,
                "--num-seqs",
                "1",
                "--min-input-len",
                "128",
                "--max-input-len",
                "128",
                "--min-output-len",
                "17",
                "--max-output-len",
                "17",
                "--warmup",
                "1",
                "--prefill-logits",
                "last",
                "--seed",
                "0",
                *([] if mode == "mlx" else ["--disable-paged-attention"]),
                *extra,
            ],
        )
        manifest["day3"].append(
            {
                "label": label,
                "mode": mode,
                "source": payload.get("source", source),
                "trace_sha256": trace_hash(payload),
            }
        )

    for sequence, mode in enumerate(
        ("seam", "inherited", "inherited", "seam"),
        start=1,
    ):
        extra = [] if mode == "seam" else ["--week3-inherit-course-projections"]
        label = f"serving-{sequence:02d}-{mode}"
        payload = run(
            label,
            [
                "benches.bench",
                "--solution",
                "ref",
                "--loader",
                "week3",
                "--model",
                MODEL,
                "--batch-decode",
                "--num-seqs",
                "8",
                "--batch-size",
                "4",
                "--min-input-len",
                "64",
                "--max-input-len",
                "256",
                "--min-output-len",
                "16",
                "--max-output-len",
                "32",
                "--prefill-step",
                "64",
                "--prefill-logits",
                "last",
                "--warmup",
                "1",
                "--seed",
                "0",
                *extra,
            ],
        )
        manifest["serving"].append(
            {
                "label": label,
                "mode": mode,
                "source": payload.get("source", source),
                "trace_sha256": trace_hash(payload),
            }
        )

    manifest_path = OUTPUT_DIR / "task360-week3-projection-e2e-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
