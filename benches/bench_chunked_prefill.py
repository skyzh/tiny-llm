import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from benches.bench_course_progression import (
    collect_host_metadata,
    collect_source_metadata,
)


@dataclass(frozen=True)
class ChunkResult:
    output_tokens_per_second: float
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    requests_per_second: float
    peak_active_requests: float
    decode_step_median_ms: float
    decode_step_p95_ms: float
    decode_step_max_ms: float
    decode_gap_median_ms: float
    decode_gap_p95_ms: float
    decode_gap_max_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare chunked-prefill budgets on one deterministic continuous-"
            "batching trace in balanced fresh processes."
        )
    )
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--solution", choices=("ref", "tiny_llm"), default="ref")
    parser.add_argument("--prefill-steps", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-input-len", type=int, default=64)
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--min-output-len", type=int, default=32)
    parser.add_argument("--max-output-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if not args.prefill_steps or any(step <= 0 for step in args.prefill_steps):
        parser.error("--prefill-steps must contain positive values")
    if len(set(args.prefill_steps)) != len(args.prefill_steps):
        parser.error("--prefill-steps must not contain duplicates")
    if args.num_seqs <= 0 or args.batch_size <= 0:
        parser.error("--num-seqs and --batch-size must be positive")
    if args.num_seqs < args.batch_size:
        parser.error("--num-seqs must be at least --batch-size")
    if args.min_input_len <= 0 or args.min_input_len > args.max_input_len:
        parser.error("invalid input-length range")
    if args.min_output_len <= 0 or args.min_output_len > args.max_output_len:
        parser.error("invalid output-length range")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    if len(args.prefill_steps) > 1 and args.repeats % 2 != 0:
        parser.error(
            "--repeats must be even when comparing chunk sizes so forward and "
            "reverse process order is balanced"
        )
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be non-negative")
    return args


def run_step(
    root: Path,
    args: argparse.Namespace,
    prefill_step: int,
    result_path: Path,
) -> tuple[ChunkResult, list[dict]]:
    command = [
        sys.executable,
        "-m",
        "benches.bench",
        "--solution",
        args.solution,
        "--loader",
        "week2",
        "--model",
        args.model,
        "--batch-decode",
        "--num-seqs",
        str(args.num_seqs),
        "--batch-size",
        str(args.batch_size),
        "--min-input-len",
        str(args.min_input_len),
        "--max-input-len",
        str(args.max_input_len),
        "--min-output-len",
        str(args.min_output_len),
        "--max-output-len",
        str(args.max_output_len),
        "--prefill-step",
        str(prefill_step),
        "--prefill-logits",
        "last",
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--json-output",
        str(result_path),
    ]
    environment = os.environ.copy()
    source_path = str(root / "src")
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not current_pythonpath
        else source_path + os.pathsep + current_pythonpath
    )
    if args.offline:
        environment["HF_HUB_OFFLINE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise subprocess.CalledProcessError(completed.returncode, command)
    payload = json.loads(result_path.read_text())
    metrics = payload["metrics"]
    result = ChunkResult(
        **{field: metrics[field] for field in ChunkResult.__dataclass_fields__}
    )
    return result, payload["request_trace"]


def median_result(samples: list[ChunkResult]) -> ChunkResult:
    return ChunkResult(
        **{
            field: statistics.median(getattr(sample, field) for sample in samples)
            for field in ChunkResult.__dataclass_fields__
        }
    )


def trace_sha256(trace: list[dict]) -> str:
    encoded = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    steps = list(args.prefill_steps)
    samples = {step: [] for step in steps}
    execution_order: list[list[int]] = []
    request_trace: list[dict] | None = None
    host = collect_host_metadata()
    print(f"Host: {host['platform']} ({host['machine']}); MLX {host['mlx_version']}")
    print(
        f"Model={args.model} requests={args.num_seqs} batch={args.batch_size} "
        f"prompt={args.min_input_len}-{args.max_input_len} "
        f"output={args.min_output_len}-{args.max_output_len} seed={args.seed} "
        f"warmup={args.warmup} repeats={args.repeats} steps={steps}"
    )

    total_runs = args.repeats * len(steps)
    completed_runs = 0
    with tempfile.TemporaryDirectory(prefix="tiny-llm-chunked-prefill-") as directory:
        temp_dir = Path(directory)
        for repeat in range(args.repeats):
            ordered = steps if repeat % 2 == 0 else list(reversed(steps))
            execution_order.append(list(ordered))
            for step in ordered:
                completed_runs += 1
                print(
                    f"[{completed_runs}/{total_runs}] prefill_step={step}",
                    file=sys.stderr,
                    flush=True,
                )
                result, current_trace = run_step(
                    root, args, step, temp_dir / f"{repeat}-{step}.json"
                )
                if request_trace is None:
                    request_trace = current_trace
                elif current_trace != request_trace:
                    raise RuntimeError(
                        "request trace changed between chunk-size samples"
                    )
                samples[step].append(result)
                if args.cooldown_seconds and completed_runs < total_runs:
                    time.sleep(args.cooldown_seconds)

    medians = {step: median_result(values) for step, values in samples.items()}
    print()
    print(
        "| Prefill step | Output tok/s | Prefill tok/s | Decode tok/s | "
        "Requests/s | Decode step p95 ms | Decode gap p95/max ms |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for step in steps:
        result = medians[step]
        print(
            f"| {step} | {result.output_tokens_per_second:.2f} | "
            f"{result.prefill_tokens_per_second:.2f} | "
            f"{result.decode_tokens_per_second:.2f} | "
            f"{result.requests_per_second:.3f} | "
            f"{result.decode_step_p95_ms:.2f} | "
            f"{result.decode_gap_p95_ms:.2f}/{result.decode_gap_max_ms:.2f} |"
        )

    if args.json_output:
        assert request_trace is not None
        payload = {
            "source": collect_source_metadata(root),
            "host": host,
            "configuration": {
                "loader": "week2",
                **{
                    key: value
                    for key, value in vars(args).items()
                    if key not in {"json_output", "prefill_steps"}
                },
            },
            "prefill_steps": steps,
            "execution_order": execution_order,
            "request_trace": request_trace,
            "request_trace_sha256": trace_sha256(request_trace),
            "samples": {
                str(step): [asdict(sample) for sample in values]
                for step, values in samples.items()
            },
            "medians": {str(step): asdict(result) for step, result in medians.items()},
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
