from itertools import permutations
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest

from benches import bench_week2_operators as benchmark


def test_public_sections_expose_current_candidates_not_retired_split_k(
    monkeypatch, capsys
):
    assert "fused-gate-up" in benchmark.SECTIONS
    monkeypatch.setattr(sys, "argv", ["bench_week2_operators.py", "--help"])

    with pytest.raises(SystemExit) as exited:
        benchmark.parse_args()

    help_text = capsys.readouterr().out
    assert exited.value.code == 0
    assert "fused-gate-up" in help_text
    assert "include-split-k" not in help_text


def test_fused_gate_up_section_compares_separate_fused_and_mlx(monkeypatch):
    class Quantized:
        weight = mx.zeros((4, 1), dtype=mx.uint32)
        scales = mx.ones((4, 1), dtype=mx.bfloat16)
        biases = mx.zeros((4, 1), dtype=mx.bfloat16)
        group_size = 128
        bits = 4

        @classmethod
        def from_mlx_layer(cls, _layer):
            return cls()

    observed = []

    def fake_benchmark(functions, warmup, iterations):
        observed.extend(name for name, _function in functions)
        assert warmup == 2
        assert iterations == 6
        return benchmark.BenchmarkComparison(
            medians_us={"separate": 3.0, "fused": 2.0, "mlx": 1.0},
            samples_us={"separate": [3.0], "fused": [2.0], "mlx": [1.0]},
            measurement_orders=[["separate", "fused", "mlx"]],
        )

    monkeypatch.setattr(benchmark, "benchmark_comparison", fake_benchmark)
    model = SimpleNamespace(
        args=SimpleNamespace(hidden_size=128),
        model=SimpleNamespace(
            embed_tokens=SimpleNamespace(scales=mx.ones((1,), dtype=mx.bfloat16)),
            layers=[
                SimpleNamespace(
                    mlp=SimpleNamespace(gate_proj=object(), up_proj=object())
                )
            ],
        ),
    )
    ops = SimpleNamespace(
        quantized_weights_type=Quantized,
        swiglu=lambda gate, up: gate * up,
        quantized_linear=lambda x, _weights: x,
        quantized_gate_up_swiglu=lambda x, _gate, _up: x,
    )

    result = benchmark.benchmark_fused_gate_up(
        SimpleNamespace(context=32, warmup=2, iterations=6), model, ops
    )

    assert observed == ["separate", "fused", "mlx"]
    assert result[0]["name"] == "gate+up+SwiGLU"


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
