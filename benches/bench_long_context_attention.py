import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import mlx.core as mx


QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
NUM_LAYERS = 36
BYTES_PER_ELEMENT = 2
DEFAULT_CONTEXTS = (2_048, 8_192, 32_768, 65_536, 131_072, 300_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MLX fused decode attention at Qwen3-4B shapes. This is "
            "an operator stress test, not a model-quality claim."
        )
    )
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS)
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def benchmark_context(context: int, warmup: int, iterations: int) -> dict:
    query = mx.random.uniform(shape=(1, QUERY_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
    keys = mx.random.uniform(shape=(1, KV_HEADS, context, HEAD_DIM)).astype(mx.bfloat16)
    values = mx.random.uniform(shape=(1, KV_HEADS, context, HEAD_DIM)).astype(
        mx.bfloat16
    )
    mx.eval(query, keys, values)

    def run() -> None:
        output = mx.fast.scaled_dot_product_attention(
            query, keys, values, scale=HEAD_DIM**-0.5
        )
        mx.eval(output)

    for _ in range(warmup):
        run()

    samples_ms = []
    for _ in range(iterations):
        start = perf_counter()
        run()
        samples_ms.append((perf_counter() - start) * 1_000)

    return {
        "context_tokens": context,
        "median_ms_per_layer": median(samples_ms),
    }


def run_worker(args: argparse.Namespace) -> None:
    results = []
    for context in args.contexts:
        results.append(benchmark_context(context, args.warmup, args.iterations))
        mx.clear_cache()
    payload = {
        "mlx_version": mx.__version__,
        "device": mx.device_info(),
        "results": results,
    }
    print(json.dumps(payload))


def run_fresh_process(args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        "-m",
        "benches.bench_long_context_attention",
        "--worker",
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--contexts",
        *(str(context) for context in args.contexts),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main() -> None:
    args = parse_args()
    if any(context <= 0 for context in args.contexts):
        raise ValueError("contexts must be positive")
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError(
            "warmup cannot be negative; iterations and repeats must be positive"
        )
    if args.worker:
        run_worker(args)
        return

    runs = [run_fresh_process(args) for _ in range(args.repeats)]
    results = []
    for index, context in enumerate(args.contexts):
        process_medians = [run["results"][index]["median_ms_per_layer"] for run in runs]
        layer_ms = median(process_medians)
        kv_bytes = NUM_LAYERS * 2 * KV_HEADS * context * HEAD_DIM * BYTES_PER_ELEMENT
        results.append(
            {
                "context_tokens": context,
                "bf16_kv_gib": kv_bytes / 2**30,
                "process_medians_ms_per_layer": process_medians,
                "median_ms_per_layer": layer_ms,
                "attention_only_decode_ceiling_tok_s": 1_000 / (NUM_LAYERS * layer_ms),
            }
        )

    payload = {
        "benchmark": "MLX fused decode attention at Qwen3-4B shapes",
        "qualification": "Synthetic operator stress test; not an end-to-end model-quality claim.",
        "mlx_version": runs[0]["mlx_version"],
        "device": runs[0]["device"],
        "shape": {
            "batch": 1,
            "query_tokens": 1,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "layers": NUM_LAYERS,
            "dtype": "bfloat16",
        },
        "method": {
            "warmup": args.warmup,
            "iterations_per_process": args.iterations,
            "fresh_processes": args.repeats,
            "statistic": "median of synchronized per-process medians",
        },
        "results": results,
    }

    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
