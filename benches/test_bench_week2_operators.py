from itertools import permutations

import pytest

from benches import bench_week2_operators as benchmark


def test_context_execution_order_balances_forward_and_reverse_sweeps():
    contexts = [32, 128, 256]

    assert benchmark.context_execution_order(contexts, 4) == [
        [32, 128, 256],
        [256, 128, 32],
        [32, 128, 256],
        [256, 128, 32],
    ]


def test_shape_execution_order_balances_context_and_query_lengths():
    assert benchmark.shape_execution_order([128], [1, 2, 4, 8], 2) == [
        [(128, 1), (128, 2), (128, 4), (128, 8)],
        [(128, 8), (128, 4), (128, 2), (128, 1)],
    ]


def test_benchmark_comparison_records_every_rotated_order(monkeypatch):
    monkeypatch.setattr(benchmark.mx, "eval", lambda value: None)
    functions = [(name, lambda name=name: name) for name in ("a", "b", "c")]

    result = benchmark.benchmark_comparison(functions, warmup=0, iterations=6)

    assert result.measurement_orders == [
        list(order) for order in permutations(("a", "b", "c"))
    ]
    assert {name: len(samples) for name, samples in result.samples_us.items()} == {
        "a": 6,
        "b": 6,
        "c": 6,
    }
    assert set(result.medians_us) == {"a", "b", "c"}


def test_benchmark_comparison_requires_complete_order_cycles():
    functions = [(name, lambda: None) for name in ("a", "b", "c")]

    with pytest.raises(ValueError, match="divisible by 6"):
        benchmark.benchmark_comparison(functions, warmup=0, iterations=5)


def test_summarize_runs_combines_raw_samples_across_context_repeats():
    runs = [
        {
            "context": 32,
            "query_length": 1,
            "sections": {
                "attention": [
                    {
                        "name": "decode attention",
                        "samples_us": {
                            "readable": [10.0, 12.0],
                            "optimized": [8.0, 9.0],
                        },
                    }
                ]
            },
        },
        {
            "context": 32,
            "query_length": 1,
            "sections": {
                "attention": [
                    {
                        "name": "decode attention",
                        "samples_us": {
                            "readable": [14.0, 16.0],
                            "optimized": [9.0, 10.0],
                        },
                    }
                ]
            },
        },
    ]

    summary = benchmark.summarize_runs(runs, [32], [1])

    assert summary == [
        {
            "context": 32,
            "query_length": 1,
            "sections": {
                "attention": [
                    {
                        "name": "decode attention",
                        "medians_us": {
                            "readable": 13.0,
                            "optimized": 9.0,
                        },
                    }
                ]
            },
        }
    ]
