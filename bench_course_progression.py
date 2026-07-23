import argparse
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    solution: str
    loader: str
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Throughput:
    prefill: float
    decode: float
    output: float


WEEK1_VARIANT = Variant("week1", "Week 1 readable", "ref", "week1")
MLX_VARIANT = Variant("mlx", "MLX", "mlx", "week2")

COURSE_VARIANTS = (
    WEEK1_VARIANT,
    Variant("week2", "Week 2 decode", "ref", "week2"),
    Variant(
        "week3-paged",
        "Week 3 paged (Flash off)",
        "ref",
        "week3",
        ("--disable-flash-attn",),
    ),
    Variant(
        "week3-flash",
        "Week 3 paged + FlashAttention",
        "ref",
        "week3",
        ("--enable-flash-attn",),
    ),
    Variant(
        "week3-lab",
        "Week 3 paged + lab",
        "ref",
        "week3",
        ("--disable-flash-attn", "--enable-performance-lab"),
    ),
    Variant(
        "week3-flash-lab",
        "Week 3 paged + Flash + lab",
        "ref",
        "week3",
        ("--enable-flash-attn", "--enable-performance-lab"),
    ),
    MLX_VARIANT,
)
WEEK2_VARIANTS = (
    WEEK1_VARIANT,
    Variant(
        "week2-kv-cache",
        "2.1 KV cache",
        "ref",
        "week2",
        ("--week2-checkpoint", "kv-cache"),
    ),
    Variant(
        "week2-quantized-matvec",
        "2.3 Quantized matvec",
        "ref",
        "week2",
        ("--week2-checkpoint", "quantized-matvec"),
    ),
    Variant(
        "week2-decode-attention",
        "2.4 Decode attention",
        "ref",
        "week2",
        ("--week2-checkpoint", "decode-attention"),
    ),
    Variant(
        "week2-rmsnorm",
        "2.5 Fast RMSNorm",
        "ref",
        "week2",
        ("--week2-checkpoint", "rmsnorm"),
    ),
    Variant(
        "week2-rope",
        "2.5 + Fast RoPE",
        "ref",
        "week2",
        ("--week2-checkpoint", "rope"),
    ),
    Variant(
        "week2-swiglu",
        "2.5 + Fused SwiGLU",
        "ref",
        "week2",
        ("--week2-checkpoint", "swiglu"),
    ),
    MLX_VARIANT,
)
VARIANTS_BY_KEY = {
    variant.key: variant for variant in (*COURSE_VARIANTS, *WEEK2_VARIANTS)
}
METRIC_PATTERN = re.compile(
    r"(Prefill|Decode|Output) throughput: ([0-9]+(?:\.[0-9]+)?) tok/s"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched tiny-llm course checkpoints in fresh sequential processes "
            "and report median throughput."
        )
    )
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument(
        "--solution",
        choices=("ref", "tiny_llm"),
        default="ref",
        help="benchmark the reference or student course checkpoints",
    )
    parser.add_argument(
        "--suite",
        choices=("course", "week2"),
        default="course",
        help="compare weekly checkpoints or the cumulative Week 2 ladder",
    )
    parser.add_argument(
        "--device",
        choices=["gpu"],
        default="gpu",
        help="execution device; the progression includes course-owned Metal kernels",
    )
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=65)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(VARIANTS_BY_KEY),
        help="run only this checkpoint; repeat the option to select several",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="set HF_HUB_OFFLINE=1; required model files must already be cached",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="pause between fresh processes when thermal stability needs it",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optionally save configuration, samples, and medians as JSON",
    )
    args = parser.parse_args()
    if args.input_len <= 0:
        parser.error("--input-len must be positive")
    if args.output_len <= 1:
        parser.error("--output-len must be greater than one to measure decode")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be non-negative")
    if args.variant:
        suite_variants = WEEK2_VARIANTS if args.suite == "week2" else COURSE_VARIANTS
        suite_keys = {variant.key for variant in suite_variants}
        invalid = [key for key in args.variant if key not in suite_keys]
        if invalid:
            parser.error(
                f"variants {invalid} do not belong to the {args.suite!r} suite"
            )
    return args


def parse_throughput(output: str) -> Throughput:
    metrics = {
        name.lower(): float(value) for name, value in METRIC_PATTERN.findall(output)
    }
    missing = {"prefill", "decode", "output"} - metrics.keys()
    if missing:
        raise ValueError(f"benchmark output is missing metrics: {sorted(missing)}")
    return Throughput(
        prefill=metrics["prefill"],
        decode=metrics["decode"],
        output=metrics["output"],
    )


def run_variant(
    root: Path,
    variant: Variant,
    args: argparse.Namespace,
) -> Throughput:
    command = [
        sys.executable,
        str(root / "bench.py"),
        "--solution",
        args.solution if variant.solution == "ref" else variant.solution,
        "--loader",
        variant.loader,
        "--model",
        args.model,
        "--device",
        args.device,
        "--num-seqs",
        "1",
        "--min-input-len",
        str(args.input_len),
        "--max-input-len",
        str(args.input_len),
        "--min-output-len",
        str(args.output_len),
        "--max-output-len",
        str(args.output_len),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        *variant.extra_args,
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
    try:
        return parse_throughput(completed.stdout)
    except ValueError:
        sys.stderr.write(completed.stdout)
        raise


def median_throughput(samples: list[Throughput]) -> Throughput:
    return Throughput(
        prefill=statistics.median(sample.prefill for sample in samples),
        decode=statistics.median(sample.decode for sample in samples),
        output=statistics.median(sample.output for sample in samples),
    )


def collect_host_metadata() -> dict:
    metadata = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "mlx_version": importlib.metadata.version("mlx"),
    }
    if sys.platform == "darwin":
        completed = subprocess.run(
            [
                "system_profiler",
                "SPHardwareDataType",
                "SPDisplaysDataType",
                "-json",
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            profile = json.loads(completed.stdout)
            hardware = profile.get("SPHardwareDataType", [{}])[0]
            display = profile.get("SPDisplaysDataType", [{}])[0]
            metadata["hardware"] = {
                "machine_name": hardware.get("machine_name"),
                "machine_model": hardware.get("machine_model"),
                "chip_type": hardware.get("chip_type"),
                "cpu_cores": hardware.get("number_processors"),
                "gpu_model": display.get("sppci_model"),
                "gpu_cores": display.get("sppci_cores"),
                "physical_memory": hardware.get("physical_memory"),
            }
    return metadata


def relative_to(value: float, baseline: float) -> str:
    if value == baseline:
        return "baseline"
    ratio = value / baseline
    if ratio < 1.0:
        return f"{(1.0 - ratio) * 100.0:.1f}% slower"
    return f"{ratio:.2f}x"


def gap_to(value: float, baseline: float) -> str:
    difference = (value / baseline - 1.0) * 100.0
    if abs(difference) < 0.05:
        return "matched"
    if difference < 0:
        return f"{-difference:.1f}% slower"
    return f"{difference:.1f}% faster"


def print_table(
    variants: list[Variant],
    medians: dict[str, Throughput],
) -> None:
    week1 = medians.get("week1")
    mlx = medians.get("mlx")
    print()
    print(
        "| Checkpoint | Prefill tok/s | vs Week 1 | vs MLX | "
        "Decode tok/s | vs Week 1 | vs MLX |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        result = medians[variant.key]
        prefill_week1 = relative_to(result.prefill, week1.prefill) if week1 else "n/a"
        prefill_mlx = gap_to(result.prefill, mlx.prefill) if mlx else "n/a"
        decode_week1 = relative_to(result.decode, week1.decode) if week1 else "n/a"
        decode_mlx = gap_to(result.decode, mlx.decode) if mlx else "n/a"
        print(
            f"| {variant.label} | {result.prefill:.2f} | {prefill_week1} | "
            f"{prefill_mlx} | {result.decode:.2f} | {decode_week1} | "
            f"{decode_mlx} |"
        )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    host = collect_host_metadata()
    variants = (
        [VARIANTS_BY_KEY[key] for key in args.variant]
        if args.variant
        else list(WEEK2_VARIANTS if args.suite == "week2" else COURSE_VARIANTS)
    )
    samples: dict[str, list[Throughput]] = {variant.key: [] for variant in variants}

    print(f"Host: {host['platform']} ({host['machine']}); MLX {host['mlx_version']}")
    print(
        f"Model={args.model} input={args.input_len} output={args.output_len} "
        f"warmup={args.warmup} repeats={args.repeats} device={args.device}"
    )
    print(
        "Run on an otherwise idle machine. Checkpoints execute sequentially in "
        "fresh processes; alternating order reduces systematic thermal bias."
    )

    completed_runs = 0
    total_runs = args.repeats * len(variants)
    for repeat in range(args.repeats):
        ordered_variants = variants if repeat % 2 == 0 else list(reversed(variants))
        for variant in ordered_variants:
            completed_runs += 1
            print(
                f"[{completed_runs}/{total_runs}] {variant.label}",
                file=sys.stderr,
                flush=True,
            )
            result = run_variant(root, variant, args)
            samples[variant.key].append(result)
            if args.cooldown_seconds and completed_runs < total_runs:
                time.sleep(args.cooldown_seconds)

    medians = {
        variant.key: median_throughput(samples[variant.key]) for variant in variants
    }
    print_table(variants, medians)

    if args.json_output:
        payload = {
            "host": host,
            "configuration": {
                "model": args.model,
                "solution": args.solution,
                "suite": args.suite,
                "device": args.device,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "seed": args.seed,
                "offline": args.offline,
                "cooldown_seconds": args.cooldown_seconds,
                "variants": [variant.key for variant in variants],
            },
            "results": {
                variant.key: {
                    "label": variant.label,
                    "samples": [asdict(sample) for sample in samples[variant.key]],
                    "median": asdict(medians[variant.key]),
                }
                for variant in variants
            },
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
