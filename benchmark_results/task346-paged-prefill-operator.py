#!/usr/bin/env python3
"""Read-only paged-prefill operator benchmark for Tiny task #347.

This lives outside the Tiny repository so the pinned source checkout stays
byte-identical.  It exercises the BF16 L>8 paged FlashAttention dispatch whose
Q/K/V staging uses Steel on the baseline head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from benches.bench_course_progression import (
    collect_host_metadata,
    collect_source_metadata,
)
from tiny_llm_ref.attention import (
    paged_attention,
    scaled_dot_product_attention_grouped,
)
from tiny_llm_ref.paged_kv_cache import TinyKvPagedCache, TinyKvPagedPool


QUERY_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 32
PREFIX_TOKENS = 64
QUERY_LENGTHS = (9, 65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--conditioning", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--shape-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.conditioning < 0
        or args.iterations <= 0
        or args.shape_repeats <= 0
    ):
        parser.error(
            "warmup/conditioning must be non-negative; iterations/repeats positive"
        )
    if args.shape_repeats % 2:
        parser.error("shape-repeats must be even for forward/reverse balance")
    return args


def random_kv(length: int) -> tuple[mx.array, mx.array]:
    shape = (1, KV_HEADS, length, HEAD_DIM)
    return (
        mx.random.normal(shape).astype(mx.bfloat16),
        mx.random.normal(shape).astype(mx.bfloat16),
    )


def build_case(query_length: int, seed: int) -> tuple[callable, dict]:
    mx.random.seed(seed)
    pool = TinyKvPagedPool(page_size=PAGE_SIZE)
    cache = TinyKvPagedCache(pool=pool)
    blocker = TinyKvPagedCache(pool=pool)
    cache.update_and_fetch(*random_kv(PREFIX_TOKENS))
    blocker.update_and_fetch(*random_kv(PAGE_SIZE))
    next_key, next_value = random_kv(query_length)
    metadata = cache.update_and_fetch_paged(
        next_key,
        next_value,
        mask="causal",
    )
    query = mx.random.normal(
        (1, QUERY_HEADS, query_length, HEAD_DIM)
    ).astype(mx.bfloat16)
    dense_key, dense_value = cache.gather_dense()

    def run() -> mx.array:
        return paged_attention(
            query,
            metadata.key_pages,
            metadata.value_pages,
            metadata.block_table,
            metadata.context_lens,
            metadata.page_size,
            scale=HEAD_DIM**-0.5,
            mask=metadata.mask,
        )

    expected = scaled_dot_product_attention_grouped(
        query,
        dense_key,
        dense_value,
        scale=HEAD_DIM**-0.5,
        mask="causal",
    )
    actual = run()
    mx.eval(expected, actual)
    max_abs_error = float(mx.max(mx.abs(actual - expected)).item())
    if not bool(mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()):
        raise AssertionError("paged prefill does not match grouped dense attention")
    correctness = {
        "max_abs_error": max_abs_error,
        "rtol": 0.02,
        "atol": 0.02,
        "context_tokens": int(metadata.context_lens.item()),
        "block_table": metadata.block_table.tolist(),
        "physical_pages_are_noncontiguous": cache.page_ids != list(
            range(cache.page_ids[0], cache.page_ids[0] + len(cache.page_ids))
        ),
    }
    return run, correctness


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    script_path = Path(__file__).resolve()
    orders = [
        list(QUERY_LENGTHS if repeat % 2 == 0 else reversed(QUERY_LENGTHS))
        for repeat in range(args.shape_repeats)
    ]
    samples: dict[str, list[float]] = {
        str(query_length): [] for query_length in QUERY_LENGTHS
    }
    correctness: dict[str, list[dict]] = {
        str(query_length): [] for query_length in QUERY_LENGTHS
    }
    # Condition both shapes before recording so first-position samples are not
    # predominantly a GPU-clock ramp. This is outside every timed region.
    for query_length in QUERY_LENGTHS:
        run, _ = build_case(query_length, args.seed + query_length)
        for _ in range(args.conditioning):
            mx.eval(run())
    runs = []
    for repeat, order in enumerate(orders):
        for position, query_length in enumerate(order):
            run, check = build_case(
                query_length,
                args.seed + repeat * 1000 + query_length,
            )
            for _ in range(args.warmup):
                mx.eval(run())
            run_samples = []
            for _ in range(args.iterations):
                started = perf_counter()
                mx.eval(run())
                run_samples.append((perf_counter() - started) * 1_000_000)
            samples[str(query_length)].extend(run_samples)
            correctness[str(query_length)].append(check)
            runs.append(
                {
                    "shape_repeat": repeat,
                    "shape_position": position,
                    "query_length": query_length,
                    "samples_us": run_samples,
                    "median_us": statistics.median(run_samples),
                    "correctness": check,
                }
            )

    payload = {
        "schema_version": 1,
        "benchmark": "Tiny required-solution paged BF16 prefill operator",
        "qualification": (
            "Synthetic synchronized operator latency; not end-to-end throughput "
            "or a production claim."
        ),
        "source": collect_source_metadata(root),
        "host": collect_host_metadata(),
        "harness": {
            "path": str(script_path),
            "sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        },
        "configuration": {
            "solution": "tiny_llm_ref",
            "operator": "paged_attention",
            "dtype": "bfloat16",
            "batch": 1,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
            "prefix_tokens": PREFIX_TOKENS,
            "query_lengths": list(QUERY_LENGTHS),
            "causal": True,
            "warmup_per_case": args.warmup,
            "untimed_conditioning_per_shape": args.conditioning,
            "iterations_per_case": args.iterations,
            "shape_repeats": args.shape_repeats,
            "shape_execution_order": orders,
            "seed": args.seed,
            "synchronization": "mx.eval(output) inside every warm-up and timed sample",
            "statistic": "median of all raw synchronized samples per query length",
        },
        "correctness": correctness,
        "runs": runs,
        "summary": {
            query_length: {
                "sample_count": len(values),
                "median_us": statistics.median(values),
                "min_us": min(values),
                "max_us": max(values),
            }
            for query_length, values in samples.items()
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
