import sys

import pytest

from benches import bench_chunked_prefill, bench_serving_progression
from benches import bench_week3_attention
from benches.bench import nearest_rank_percentile, sample_median


def test_latency_statistics_use_explicit_nearest_rank():
    samples = [4.0, 1.0, 3.0, 2.0]

    assert sample_median(samples) == 2.5
    assert nearest_rank_percentile(samples, 0.50) == 2.0
    assert nearest_rank_percentile(samples, 0.95) == 4.0
    assert nearest_rank_percentile([], 0.95) == 0.0
    with pytest.raises(ValueError, match="quantile"):
        nearest_rank_percentile(samples, 0.0)


def test_serving_comparison_rejects_odd_process_order(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench-serving-progression", "--repeats", "3"])

    with pytest.raises(SystemExit):
        bench_serving_progression.parse_args()


def test_single_serving_variant_allows_odd_repeats(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench-serving-progression",
            "--variant",
            "paged",
            "--repeats",
            "3",
        ],
    )

    assert bench_serving_progression.parse_args().repeats == 3


def test_chunk_comparison_rejects_odd_process_order(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench-chunked-prefill", "--repeats", "3"])

    with pytest.raises(SystemExit):
        bench_chunked_prefill.parse_args()


def test_request_trace_checksum_is_canonical():
    first = [{"request_id": 0, "prompt_token_ids": [3, 1], "max_new_tokens": 4}]
    reordered = [{"max_new_tokens": 4, "prompt_token_ids": [3, 1], "request_id": 0}]
    changed = [{"request_id": 0, "prompt_token_ids": [3, 2], "max_new_tokens": 4}]

    assert bench_chunked_prefill.trace_sha256(first) == (
        bench_chunked_prefill.trace_sha256(reordered)
    )
    assert bench_chunked_prefill.trace_sha256(first) != (
        bench_chunked_prefill.trace_sha256(changed)
    )
    assert bench_serving_progression.trace_sha256(first) == (
        bench_chunked_prefill.trace_sha256(first)
    )


def test_week3_operator_comparison_rejects_odd_process_order(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench-week3-attention", "--repeats", "3"])

    with pytest.raises(SystemExit):
        bench_week3_attention.parse_args()
