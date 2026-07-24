import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from bench_course_progression import collect_host_metadata


@dataclass(frozen=True)
class ServingVariant:
    key: str
    label: str
    loader: str


@dataclass(frozen=True)
class ServingResult:
    output_tokens_per_second: float
    total_tokens_per_second: float
    decode_tokens_per_second: float
    requests_per_second: float
    peak_kv_bytes: float
    peak_active_requests: float
    peak_live_pages: float
    peak_capacity_pages: float
    peak_tail_waste_slots: float
    reused_page_allocations: float
    storage_growths: float
    copied_pages_on_growth: float
    paged_growth_copy_bytes: float
    dense_growth_copy_bytes: float
    dense_staging_copy_bytes: float


VARIANTS = (
    ServingVariant("dense", "Week 2 dense KV", "week2"),
    ServingVariant("paged", "Week 3 paged KV", "week3"),
)
VARIANTS_BY_KEY = {variant.key: variant for variant in VARIANTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched continuous-batching workloads with dynamically growing "
            "KV caches in fresh processes."
        )
    )
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--solution", choices=("ref", "tiny_llm"), default="ref")
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-input-len", type=int, default=128)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=32)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--prefill-step", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(VARIANTS_BY_KEY),
        help="run only this storage path; repeat to select both",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.num_seqs <= 0 or args.batch_size <= 0:
        parser.error("--num-seqs and --batch-size must be positive")
    if args.num_seqs < args.batch_size:
        parser.error("--num-seqs must be at least --batch-size")
    if args.min_input_len <= 0 or args.min_input_len > args.max_input_len:
        parser.error("invalid input-length range")
    if args.min_output_len <= 0 or args.min_output_len > args.max_output_len:
        parser.error("invalid output-length range")
    if args.prefill_step <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("invalid prefill, warmup, or repeat count")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be non-negative")
    return args


def run_variant(
    root: Path,
    variant: ServingVariant,
    args: argparse.Namespace,
    result_path: Path,
) -> ServingResult:
    command = [
        sys.executable,
        str(root / "bench.py"),
        "--solution",
        args.solution,
        "--loader",
        variant.loader,
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
        str(args.prefill_step),
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
    metrics = json.loads(result_path.read_text())["metrics"]
    return ServingResult(
        **{field: metrics[field] for field in ServingResult.__dataclass_fields__}
    )


def median_result(samples: list[ServingResult]) -> ServingResult:
    return ServingResult(
        **{
            field: statistics.median(getattr(sample, field) for sample in samples)
            for field in ServingResult.__dataclass_fields__
        }
    )


def format_mib(value: float) -> str:
    return f"{value / (1024 * 1024):.1f}"


def print_table(
    variants: list[ServingVariant], medians: dict[str, ServingResult]
) -> None:
    print()
    print(
        "| Storage path | Output tok/s | Decode tok/s | Requests/s | "
        "Peak KV MiB | Avoidable KV copy MiB |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for variant in variants:
        result = medians[variant.key]
        copied_bytes = (
            result.dense_growth_copy_bytes + result.dense_staging_copy_bytes
            if variant.key == "dense"
            else result.paged_growth_copy_bytes
        )
        print(
            f"| {variant.label} | {result.output_tokens_per_second:.2f} | "
            f"{result.decode_tokens_per_second:.2f} | "
            f"{result.requests_per_second:.2f} | "
            f"{format_mib(result.peak_kv_bytes)} | {format_mib(copied_bytes)} |"
        )
    paged = medians.get("paged")
    if paged:
        print()
        print(
            "Paged allocator: "
            f"peak_live_pages={paged.peak_live_pages:.0f}, "
            f"peak_capacity_pages={paged.peak_capacity_pages:.0f}, "
            f"peak_tail_waste_slots={paged.peak_tail_waste_slots:.0f}, "
            f"reused_allocations={paged.reused_page_allocations:.0f}, "
            f"pool_growths={paged.storage_growths:.0f}."
        )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    variants = (
        [VARIANTS_BY_KEY[key] for key in args.variant]
        if args.variant
        else list(VARIANTS)
    )
    host = collect_host_metadata()
    samples = {variant.key: [] for variant in variants}
    print(f"Host: {host['platform']} ({host['machine']}); MLX {host['mlx_version']}")
    print(
        f"Model={args.model} requests={args.num_seqs} batch={args.batch_size} "
        f"prompt={args.min_input_len}-{args.max_input_len} "
        f"output={args.min_output_len}-{args.max_output_len} "
        f"prefill_step={args.prefill_step} warmup={args.warmup} "
        f"repeats={args.repeats}"
    )
    print(
        "The measured run resets page pools after warmup; no request context "
        "capacity is preallocated."
    )

    total_runs = args.repeats * len(variants)
    completed_runs = 0
    with tempfile.TemporaryDirectory(prefix="tiny-llm-serving-") as directory:
        temp_dir = Path(directory)
        for repeat in range(args.repeats):
            ordered = variants if repeat % 2 == 0 else list(reversed(variants))
            for variant in ordered:
                completed_runs += 1
                print(
                    f"[{completed_runs}/{total_runs}] {variant.label}",
                    file=sys.stderr,
                    flush=True,
                )
                result_path = temp_dir / f"{repeat}-{variant.key}.json"
                samples[variant.key].append(
                    run_variant(root, variant, args, result_path)
                )
                if args.cooldown_seconds and completed_runs < total_runs:
                    time.sleep(args.cooldown_seconds)

    medians = {variant.key: median_result(samples[variant.key]) for variant in variants}
    print_table(variants, medians)
    if args.json_output:
        payload = {
            "host": host,
            "configuration": {
                key: value
                for key, value in vars(args).items()
                if key not in {"json_output", "variant"}
            },
            "samples": {
                key: [asdict(sample) for sample in value]
                for key, value in samples.items()
            },
            "medians": {key: asdict(value) for key, value in medians.items()},
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
