import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from benches.bench_course_progression import (
    collect_host_metadata,
    collect_source_metadata,
)


QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
VARIANTS = ("dense-gather", "paged", "mlx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark matched Qwen3 paged-decode operator paths in balanced "
            "fresh processes."
        )
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.contexts or any(context < 2 for context in args.contexts):
        parser.error("--contexts must contain values >= 2")
    if len(set(args.contexts)) != len(args.contexts):
        parser.error("--contexts must not contain duplicates")
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error(
            "--warmup must be non-negative; --iterations and --repeats must be positive"
        )
    selected_variant_count = len(args.variant) if args.variant else len(VARIANTS)
    if selected_variant_count > 1 and args.repeats % 2 != 0:
        parser.error(
            "--repeats must be even when comparing variants so forward and "
            "reverse process order is balanced"
        )
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be non-negative")
    if args.worker and (args.variant is None or len(args.variant) != 1):
        parser.error("--worker requires exactly one --variant")
    return args


def build_case(context: int, page_size: int, seed: int) -> dict:
    from tiny_llm_ref.attention import (
        paged_attention,
        scaled_dot_product_attention_grouped,
    )
    from tiny_llm_ref.paged_kv_cache import TinyKvPagedCache, TinyKvPagedPool

    mx.random.seed(seed + context)
    query = mx.random.normal((1, QUERY_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
    keys = mx.random.normal((1, KV_HEADS, context, HEAD_DIM)).astype(mx.bfloat16)
    values = mx.random.normal((1, KV_HEADS, context, HEAD_DIM)).astype(mx.bfloat16)
    pool = TinyKvPagedPool(page_size=page_size)
    cache = TinyKvPagedCache(pool=pool)
    cache.update_and_fetch(keys[:, :, :-1, :], values[:, :, :-1, :])
    metadata = cache.update_and_fetch_paged(keys[:, :, -1:, :], values[:, :, -1:, :])
    dense_keys, dense_values = cache.gather_dense()
    mx.eval(query, metadata.key_pages, metadata.value_pages, dense_keys, dense_values)

    def dense_gather() -> mx.array:
        gathered_keys, gathered_values = cache.gather_dense()
        return scaled_dot_product_attention_grouped(
            query,
            gathered_keys,
            gathered_values,
            scale=HEAD_DIM**-0.5,
        )

    def direct_paged() -> mx.array:
        return paged_attention(
            query,
            metadata.key_pages,
            metadata.value_pages,
            metadata.block_table,
            metadata.context_lens,
            metadata.page_size,
            scale=HEAD_DIM**-0.5,
        )

    def mlx_fused() -> mx.array:
        return mx.fast.scaled_dot_product_attention(
            query,
            dense_keys,
            dense_values,
            scale=HEAD_DIM**-0.5,
        )

    expected = dense_gather()
    direct = direct_paged()
    mlx_output = mlx_fused()
    mx.eval(expected, direct, mlx_output)
    direct_max_abs_error = float(mx.max(mx.abs(direct - expected)).item())
    mlx_max_abs_error = float(mx.max(mx.abs(mlx_output - expected)).item())
    if not bool(mx.allclose(direct, expected, rtol=2e-2, atol=2e-2).item()):
        raise AssertionError("direct paged attention does not match dense attention")
    if not bool(mx.allclose(mlx_output, expected, rtol=2e-2, atol=2e-2).item()):
        raise AssertionError("MLX attention does not match dense attention")
    return {
        "functions": {
            "dense-gather": dense_gather,
            "paged": direct_paged,
            "mlx": mlx_fused,
        },
        "direct_max_abs_error": direct_max_abs_error,
        "mlx_max_abs_error": mlx_max_abs_error,
    }


def benchmark_variant(args: argparse.Namespace, variant: str) -> dict:
    results = []
    for context in args.contexts:
        case = build_case(context, args.page_size, args.seed)
        run = case["functions"][variant]
        for _ in range(args.warmup):
            mx.eval(run())
        samples_us = []
        for _ in range(args.iterations):
            started = perf_counter()
            mx.eval(run())
            samples_us.append((perf_counter() - started) * 1_000_000)
        results.append(
            {
                "context_tokens": context,
                "samples_us": samples_us,
                "median_us": statistics.median(samples_us),
                "direct_max_abs_error": case["direct_max_abs_error"],
                "mlx_max_abs_error": case["mlx_max_abs_error"],
            }
        )
        mx.clear_cache()
    return {"variant": variant, "mlx_version": mx.__version__, "results": results}


def run_worker(args: argparse.Namespace) -> None:
    assert args.variant is not None
    print(json.dumps(benchmark_variant(args, args.variant[0])))


def run_fresh_process(args: argparse.Namespace, variant: str) -> dict:
    command = [
        sys.executable,
        "-m",
        "benches.bench_week3_attention",
        "--worker",
        "--variant",
        variant,
        "--contexts",
        *(str(context) for context in args.contexts),
        "--page-size",
        str(args.page_size),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
    ]
    environment = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
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
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return

    variants = args.variant if args.variant else list(VARIANTS)
    host = collect_host_metadata()
    samples = {
        variant: {context: [] for context in args.contexts} for variant in variants
    }
    correctness: dict[str, dict[str, float]] = {}
    execution_order: list[list[str]] = []
    total_runs = args.repeats * len(variants)
    completed_runs = 0
    for repeat in range(args.repeats):
        ordered = variants if repeat % 2 == 0 else list(reversed(variants))
        execution_order.append(list(ordered))
        for variant in ordered:
            completed_runs += 1
            print(
                f"[{completed_runs}/{total_runs}] {variant}",
                file=sys.stderr,
                flush=True,
            )
            run = run_fresh_process(args, variant)
            for result in run["results"]:
                context = result["context_tokens"]
                samples[variant][context].append(result)
                correctness[str(context)] = {
                    "direct_max_abs_error": result["direct_max_abs_error"],
                    "mlx_max_abs_error": result["mlx_max_abs_error"],
                }
            if args.cooldown_seconds and completed_runs < total_runs:
                time.sleep(args.cooldown_seconds)

    medians = {
        variant: {
            str(context): statistics.median(
                result["median_us"] for result in samples[variant][context]
            )
            for context in args.contexts
        }
        for variant in variants
    }
    print(f"Host: {host['platform']} ({host['machine']}); MLX {host['mlx_version']}")
    print(
        f"Shape=B1,Hq{QUERY_HEADS},Hkv{KV_HEADS},L1,D{HEAD_DIM},BF16 "
        f"page_size={args.page_size} warmup={args.warmup} "
        f"iterations={args.iterations} repeats={args.repeats}"
    )
    print()
    labels = {
        "dense-gather": "Dense + gather us",
        "paged": "Direct paged us",
        "mlx": "MLX fused us",
    }
    print("| Context | " + " | ".join(labels[variant] for variant in variants) + " |")
    print("|---:|" + "---:|" * len(variants))
    for context in args.contexts:
        values = [medians[variant][str(context)] for variant in variants]
        print(f"| {context} | " + " | ".join(f"{value:.2f}" for value in values) + " |")

    if args.json_output:
        payload = {
            "source": collect_source_metadata(Path(__file__).resolve().parents[1]),
            "host": host,
            "configuration": {
                "model_shape": "qwen3-4b",
                "solution": "tiny_llm_ref",
                "operator": "paged_attention",
                "contexts": args.contexts,
                "page_size": args.page_size,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "seed": args.seed,
                "offline": args.offline,
                "cooldown_seconds": args.cooldown_seconds,
                "dtype": "bfloat16",
                "batch": 1,
                "query_heads": QUERY_HEADS,
                "kv_heads": KV_HEADS,
                "query_tokens": 1,
                "head_dim": HEAD_DIM,
                "scale": HEAD_DIM**-0.5,
                "variants": variants,
            },
            "execution_order": execution_order,
            "correctness": correctness,
            "samples": {
                variant: {
                    str(context): results for context, results in by_context.items()
                }
                for variant, by_context in samples.items()
            },
            "medians_us": medians,
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
