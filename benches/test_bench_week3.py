import subprocess
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


def test_week3_operator_solution_defaults_to_reference(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench-week3-attention"])

    assert bench_week3_attention.parse_args().solution == "ref"


@pytest.mark.parametrize("solution", ["ref", "tiny_llm"])
def test_week3_operator_accepts_public_solution_choices(monkeypatch, solution):
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench-week3-attention", "--solution", solution],
    )

    assert bench_week3_attention.parse_args().solution == solution


def test_week3_operator_rejects_unknown_solution(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench-week3-attention", "--solution", "unknown"],
    )

    with pytest.raises(SystemExit):
        bench_week3_attention.parse_args()


@pytest.mark.parametrize(
    ("solution", "package"),
    [("ref", "tiny_llm_ref"), ("tiny_llm", "tiny_llm")],
)
def test_week3_operator_imports_both_surfaces_from_selected_package(solution, package):
    attention, paged_kv_cache = bench_week3_attention.load_solution_surfaces(solution)

    assert attention.__package__ == package
    assert paged_kv_cache.__package__ == package


@pytest.mark.parametrize("solution", ["ref", "tiny_llm"])
def test_week3_operator_propagates_solution_once_to_worker(monkeypatch, solution):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench-week3-attention",
            "--solution",
            solution,
            "--variant",
            "paged",
            "--repeats",
            "1",
        ],
    )
    args = bench_week3_attention.parse_args()
    captured = []

    def run(command, **_kwargs):
        captured.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout='{"results": []}')

    monkeypatch.setattr(subprocess, "run", run)

    assert bench_week3_attention.run_fresh_process(args, "paged") == {"results": []}
    assert captured.count("--solution") == 1
    assert captured[captured.index("--solution") + 1] == solution


@pytest.mark.parametrize(
    ("solution", "package"),
    [("ref", "tiny_llm_ref"), ("tiny_llm", "tiny_llm")],
)
def test_week3_operator_json_attribution_matches_selected_solution(
    monkeypatch, solution, package
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench-week3-attention", "--solution", solution],
    )
    args = bench_week3_attention.parse_args()

    configuration = bench_week3_attention.result_configuration(args, ["paged"])

    assert configuration["solution"] == package
