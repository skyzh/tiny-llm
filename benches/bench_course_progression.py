import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
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
    dispatch_counters: dict[str, int] | None = None


@dataclass(frozen=True)
class ProductMetrics:
    TTFT_ms: float
    prefill_tokens_per_second: float
    TPOT_ms: float
    decode_tokens_per_second: float
    output_tokens_per_second: float


WEEK1_VARIANT = Variant("week1", "Week 1 readable", "ref", "week1")
MLX_VARIANT = Variant("mlx", "MLX", "mlx", "week2")

COURSE_VARIANTS = (
    WEEK1_VARIANT,
    Variant("week2", "Week 2 decode", "ref", "week2"),
    Variant("week3", "Week 3 paged FlashAttention", "ref", "week3"),
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
        "week2-rmsnorm",
        "2.4 Fast RMSNorm",
        "ref",
        "week2",
        ("--week2-checkpoint", "rmsnorm"),
    ),
    Variant(
        "week2-rope",
        "2.4 + Fast RoPE",
        "ref",
        "week2",
        ("--week2-checkpoint", "rope"),
    ),
    Variant(
        "week2-swiglu",
        "2.4 + Fused SwiGLU",
        "ref",
        "week2",
        ("--week2-checkpoint", "swiglu"),
    ),
    Variant(
        "week2-simd-matmul",
        "2.5 SIMD matrix prefill",
        "ref",
        "week2",
        ("--week2-checkpoint", "simd-matmul"),
    ),
    Variant(
        "week2-context-selected-attention",
        "2.6 Context-selected decode attention",
        "ref",
        "week2",
        ("--week2-checkpoint", "context-selected-attention"),
    ),
    Variant(
        "week2-prefill-fused-gate-up",
        "2.7 Prefill-only fused gate+up SwiGLU",
        "ref",
        "week2",
        ("--week2-checkpoint", "prefill-fused-gate-up"),
    ),
    MLX_VARIANT,
)
VARIANTS_BY_KEY = {
    variant.key: variant for variant in (*COURSE_VARIANTS, *WEEK2_VARIANTS)
}
METRIC_PATTERN = re.compile(
    r"(Prefill|Decode|Output) throughput: ([0-9]+(?:\.[0-9]+)?) tok/s"
)
MATRIX_PROMPT_LENGTHS = (128, 512, 2048, 8192, 32640)
MATRIX_DEFAULT_PROMPT_LENGTHS = (128, 512, 2048, 8192)
MATRIX_DEFAULT_VARIANTS = ("week2-simd-matmul", "mlx")
MATRIX_OUTPUT_TOKENS = 128
NATIVE_CONTEXT_TOKENS = 32768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched tiny-llm course checkpoints in fresh sequential processes "
            "and report median throughput. Matrix mode reports TTFT_ms, "
            "prefill_tokens_per_second, TPOT_ms, and decode_tokens_per_second."
        )
    )
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument(
        "--solution",
        choices=("ref", "tiny_llm"),
        required=True,
        help="benchmark the reference or learner course checkpoints explicitly",
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
    parser.add_argument("--output-len", type=int)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help=(
            "run a Week 2 prompt-length matrix; paired Day 5/MLX defaults are "
            "128, 512, 2048, and 8192. 32640 is available only with --variant "
            "mlx at the native 32768-token ceiling"
        ),
    )
    parser.add_argument(
        "--prompt-length",
        action="append",
        type=int,
        choices=MATRIX_PROMPT_LENGTHS,
        help=(
            "matrix prompt length; repeat to select a subset of the supported "
            "128/512/2048/8192 paired points or the MLX-only 32640 endpoint"
        ),
    )
    parser.add_argument(
        "--disable-week2-context-selected-attention",
        action="store_true",
        help="disable only the Week 2 context-selected attention candidate",
    )
    parser.add_argument(
        "--disable-week2-prefill-fused-gate-up",
        action="store_true",
        help="disable only the Week 2 prefill fused gate+up candidate",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="fresh-process samples; comparisons require an even count",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefill-logits",
        choices=("all", "last"),
        default="all",
        help=(
            "compute all prompt logits for the course progression or only "
            "the final row for a serving comparison"
        ),
    )
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
    if args.output_len is None:
        args.output_len = MATRIX_OUTPUT_TOKENS if args.matrix else 65
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
    if args.prompt_length and not args.matrix:
        parser.error("--prompt-length requires --matrix")
    if args.matrix and args.suite != "week2":
        parser.error("--matrix requires --suite week2")
    if args.matrix and args.input_len != 128:
        parser.error(
            "--input-len is single-point only; use --prompt-length with --matrix"
        )
    if args.matrix and args.prefill_logits != "last":
        parser.error("--matrix requires --prefill-logits last for product inference")
    if args.matrix and args.output_len != MATRIX_OUTPUT_TOKENS:
        parser.error("--matrix requires exactly --output-len 128")
    if args.variant:
        suite_variants = WEEK2_VARIANTS if args.suite == "week2" else COURSE_VARIANTS
        suite_keys = {variant.key for variant in suite_variants}
        invalid = [key for key in args.variant if key not in suite_keys]
        if invalid:
            parser.error(
                f"variants {invalid} do not belong to the {args.suite!r} suite"
            )
    if args.variant:
        selected_variants = [VARIANTS_BY_KEY[key] for key in args.variant]
    elif args.matrix:
        selected_variants = [VARIANTS_BY_KEY[key] for key in MATRIX_DEFAULT_VARIANTS]
    else:
        selected_variants = list(
            WEEK2_VARIANTS if args.suite == "week2" else COURSE_VARIANTS
        )
        if args.prefill_logits == "last":
            selected_variants = [
                variant for variant in selected_variants if variant.loader != "week1"
            ]
    if len(selected_variants) > 1 and args.repeats % 2 != 0:
        parser.error(
            "--repeats must be even when comparing variants so forward and "
            "reverse execution orders are balanced"
        )
    if args.prefill_logits == "last" and any(
        variant.loader == "week1" for variant in selected_variants
    ):
        parser.error("--prefill-logits last requires variants that exclude Week 1")
    if args.matrix:
        args.prompt_length = list(
            dict.fromkeys(args.prompt_length or MATRIX_DEFAULT_PROMPT_LENGTHS)
        )
        has_native_endpoint = 32640 in args.prompt_length
        if has_native_endpoint and any(
            variant.key != "mlx" for variant in selected_variants
        ):
            parser.error(
                "matrix prompt length 32640 is available only for MLX; "
                "select only --variant mlx"
            )
        is_mlx_only_endpoint = has_native_endpoint and [
            variant.key for variant in selected_variants
        ] == ["mlx"]
        if len(selected_variants) != 2 and not is_mlx_only_endpoint:
            parser.error(
                "--matrix requires exactly two Week 2 variants: a course/MLX "
                "baseline pair or two course checkpoints; the sole exception "
                "is --variant mlx with --prompt-length 32640"
            )
    else:
        args.prompt_length = [args.input_len]
    args.variant = [variant.key for variant in selected_variants]
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
    input_len: int | None = None,
) -> Throughput:
    effective_input_len = args.input_len if input_len is None else input_len
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

    with tempfile.TemporaryDirectory(prefix=".bench-progression-", dir=root) as temp:
        raw_output = Path(temp) / "sample.json"
        command = [
            sys.executable,
            "-m",
            "benches.bench",
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
            str(effective_input_len),
            "--max-input-len",
            str(effective_input_len),
            "--min-output-len",
            str(args.output_len),
            "--max-output-len",
            str(args.output_len),
            "--warmup",
            str(args.warmup),
            "--prefill-logits",
            args.prefill_logits,
            "--seed",
            str(args.seed),
            "--json-output",
            str(raw_output),
            *(
                ["--disable-week2-context-selected-attention"]
                if getattr(args, "disable_week2_context_selected_attention", False)
                and variant.solution != "mlx"
                else []
            ),
            *(
                ["--disable-week2-prefill-fused-gate-up"]
                if getattr(args, "disable_week2_prefill_fused_gate_up", False)
                and variant.solution != "mlx"
                else []
            ),
            *variant.extra_args,
        ]
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
            payload = json.loads(raw_output.read_text())
            metrics = payload["metrics"]
            return Throughput(
                prefill=float(metrics["prefill_tokens_per_second"]),
                decode=float(metrics["decode_tokens_per_second"]),
                output=float(metrics["output_tokens_per_second"]),
                dispatch_counters={
                    str(name): int(count)
                    for name, count in payload.get("dispatch_counters", {}).items()
                },
            )
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            sys.stderr.write(completed.stdout)
            raise


def median_throughput(samples: list[Throughput]) -> Throughput:
    return Throughput(
        prefill=statistics.median(sample.prefill for sample in samples),
        decode=statistics.median(sample.decode for sample in samples),
        output=statistics.median(sample.output for sample in samples),
    )


def product_metrics(sample: Throughput, prompt_tokens: int) -> ProductMetrics:
    if min(sample.prefill, sample.decode, sample.output) <= 0:
        raise ValueError("product throughput metrics must be positive")
    return ProductMetrics(
        TTFT_ms=prompt_tokens / sample.prefill * 1_000.0,
        prefill_tokens_per_second=sample.prefill,
        TPOT_ms=1_000.0 / sample.decode,
        decode_tokens_per_second=sample.decode,
        output_tokens_per_second=sample.output,
    )


def summarize_product_metrics(
    samples: list[ProductMetrics],
) -> dict[str, dict[str, float]]:
    if not samples:
        raise ValueError("cannot summarize an empty product sample set")
    return {
        field: {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "spread": max(values) - min(values),
        }
        for field in ProductMetrics.__dataclass_fields__
        for values in ([getattr(sample, field) for sample in samples],)
    }


def prompt_classification(prompt_tokens: int) -> str:
    if prompt_tokens == 128:
        return "micro/regression"
    if prompt_tokens == 32640:
        return "native-product-endpoint"
    return "product-context"


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


def collect_source_metadata(root: Path) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "tree": subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "tracked_dirty": bool(tracked_status),
    }


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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


def print_matrix_table(
    prompt_lengths: list[int],
    variants: list[Variant],
    summaries: dict[int, dict[str, dict[str, dict[str, float]]]],
) -> None:
    print()
    print(
        "| Prompt | Class | Checkpoint | TTFT ms | Prefill tok/s | "
        "TPOT ms | Decode tok/s |"
    )
    print("|---:|---|---|---:|---:|---:|---:|")
    for prompt_tokens in prompt_lengths:
        for variant in variants:
            metrics = summaries[prompt_tokens][variant.key]
            print(
                f"| {prompt_tokens} | {prompt_classification(prompt_tokens)} | "
                f"{variant.label} | {metrics['TTFT_ms']['median']:.3f} | "
                f"{metrics['prefill_tokens_per_second']['median']:.2f} | "
                f"{metrics['TPOT_ms']['median']:.3f} | "
                f"{metrics['decode_tokens_per_second']['median']:.2f} |"
            )


def main() -> None:
    args = parse_args()
    if args.json_output is not None and (
        args.json_output.exists() or args.json_output.is_symlink()
    ):
        raise FileExistsError(f"refusing to overwrite {args.json_output}")
    root = Path(__file__).resolve().parents[1]
    host = collect_host_metadata()
    variants = [VARIANTS_BY_KEY[key] for key in args.variant]
    prompt_lengths = args.prompt_length
    samples: dict[int, dict[str, list[Throughput]]] = {
        prompt_tokens: {variant.key: [] for variant in variants}
        for prompt_tokens in prompt_lengths
    }
    execution_order: list[dict[str, object]] = []

    print(f"Host: {host['platform']} ({host['machine']}); MLX {host['mlx_version']}")
    print(
        f"Model={args.model} input={'matrix' if args.matrix else args.input_len} "
        f"output={args.output_len} "
        f"warmup={args.warmup} repeats={args.repeats} device={args.device} "
        f"prefill_logits={args.prefill_logits}"
    )
    print(
        "Run on an otherwise idle machine. Checkpoints execute sequentially in "
        "fresh processes; alternating order reduces systematic thermal bias."
    )

    if args.matrix:
        print(
            "Matrix points: 128 is a micro/regression point; paired course/MLX "
            "rows stop at 8192. The MLX-only 32640 prompt + 128 generated tokens "
            "reaches the native 32768-token ceiling. A 32768-token prompt is "
            "prefill-only, never a native generation row. This matrix makes no "
            "100K+ product claim."
        )

    completed_runs = 0
    total_runs = args.repeats * len(prompt_lengths) * len(variants)
    for repeat in range(args.repeats):
        ordered_lengths = (
            prompt_lengths if repeat % 2 == 0 else list(reversed(prompt_lengths))
        )
        for length_index, prompt_tokens in enumerate(ordered_lengths):
            ordered_variants = (
                variants
                if (repeat + length_index) % 2 == 0
                else list(reversed(variants))
            )
            execution_order.append(
                {
                    "repeat": repeat,
                    "prompt_tokens": prompt_tokens,
                    "variants": [variant.key for variant in ordered_variants],
                }
            )
            for variant in ordered_variants:
                completed_runs += 1
                print(
                    f"[{completed_runs}/{total_runs}] prompt={prompt_tokens} "
                    f"{variant.label}",
                    file=sys.stderr,
                    flush=True,
                )
                result = run_variant(root, variant, args, prompt_tokens)
                samples[prompt_tokens][variant.key].append(result)
                if args.cooldown_seconds and completed_runs < total_runs:
                    time.sleep(args.cooldown_seconds)

    medians = {
        prompt_tokens: {
            variant.key: median_throughput(samples[prompt_tokens][variant.key])
            for variant in variants
        }
        for prompt_tokens in prompt_lengths
    }
    product_samples = {
        prompt_tokens: {
            variant.key: [
                product_metrics(sample, prompt_tokens)
                for sample in samples[prompt_tokens][variant.key]
            ]
            for variant in variants
        }
        for prompt_tokens in prompt_lengths
    }
    summaries = {
        prompt_tokens: {
            variant.key: summarize_product_metrics(
                product_samples[prompt_tokens][variant.key]
            )
            for variant in variants
        }
        for prompt_tokens in prompt_lengths
    }
    if args.matrix:
        print_matrix_table(prompt_lengths, variants, summaries)
    else:
        print_table(variants, medians[prompt_lengths[0]])

    if args.json_output:
        workload = {
            "model": args.model,
            "input_tokens": args.input_len,
            "output_tokens": args.output_len,
            "seed": args.seed,
            "prompt_rule": "synthetic-token-ids",
            "prefill_logits": args.prefill_logits,
            "warmup": args.warmup,
            "repeats": args.repeats,
        }
        configuration = {
            "model": args.model,
            "solution": args.solution,
            "suite": args.suite,
            "device": args.device,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "prefill_logits": args.prefill_logits,
            "offline": args.offline,
            "cooldown_seconds": args.cooldown_seconds,
            "variants": [variant.key for variant in variants],
        }
        if args.matrix:
            workload.pop("input_tokens")
            workload.update(
                prompt_lengths=prompt_lengths,
                decode_sample_tokens=args.output_len - 1,
            )
            configuration.update(
                matrix=True,
                prompt_lengths=prompt_lengths,
                disable_week2_context_selected_attention=getattr(
                    args, "disable_week2_context_selected_attention", False
                ),
                disable_week2_prefill_fused_gate_up=getattr(
                    args, "disable_week2_prefill_fused_gate_up", False
                ),
            )
        payload = {
            "schema_version": 3 if args.matrix else 2,
            "source": collect_source_metadata(root),
            "host": host,
            "workload": workload,
            "workload_id": canonical_hash(workload),
            "configuration": configuration,
            "execution_order": execution_order,
        }
        if args.matrix:
            payload.update(
                evidence_kind="single_request_product_matrix",
                process_isolation="fresh_process_per_sample",
                context_boundary=(
                    "128 is micro/regression; paired course/MLX rows stop at "
                    "8192; the MLX-only 32640 prompt + 128 generated tokens "
                    "reaches the native 32768-token ceiling; a 32768 prompt is "
                    "prefill-only; no 100K+ product claim"
                ),
                generated_tokens=MATRIX_OUTPUT_TOKENS,
                post_first_decode_intervals=MATRIX_OUTPUT_TOKENS - 1,
                native_context_tokens=NATIVE_CONTEXT_TOKENS,
                metric_definitions={
                    "TTFT_ms": "prefill timer through generated token 1 of 128",
                    "prefill_tokens_per_second": "prompt tokens divided by prefill time",
                    "TPOT_ms": "decode time divided by 127 post-first-token intervals",
                    "decode_tokens_per_second": (
                        "127 post-first-token intervals divided by decode time"
                    ),
                    "output_tokens_per_second": (
                        "all generated tokens divided by complete-request elapsed time"
                    ),
                },
                prompt_classification={
                    str(length): prompt_classification(length)
                    for length in prompt_lengths
                },
                results=[
                    {
                        "prompt_tokens": prompt_tokens,
                        "prompt_classification": prompt_classification(prompt_tokens),
                        "variant": variant.key,
                        "label": variant.label,
                        "samples": [
                            {
                                **asdict(sample),
                                "dispatch_counters": (
                                    samples[prompt_tokens][variant.key][
                                        index
                                    ].dispatch_counters
                                    or {}
                                ),
                            }
                            for index, sample in enumerate(
                                product_samples[prompt_tokens][variant.key]
                            )
                        ],
                        "summary": summaries[prompt_tokens][variant.key],
                    }
                    for prompt_tokens in prompt_lengths
                    for variant in variants
                ],
            )
        else:
            prompt_tokens = prompt_lengths[0]
            payload["execution_order"] = [
                entry["variants"] for entry in execution_order
            ]
            payload["results"] = {
                variant.key: {
                    "label": variant.label,
                    "samples": [
                        asdict(sample) for sample in samples[prompt_tokens][variant.key]
                    ],
                    "median": {
                        key: value
                        for key, value in asdict(
                            medians[prompt_tokens][variant.key]
                        ).items()
                        if key != "dispatch_counters"
                    },
                }
                for variant in variants
            }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
