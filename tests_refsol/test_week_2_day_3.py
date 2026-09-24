"""Week 2 Day 3 SIMD-matrix prefill tests."""

import importlib
import sys

import mlx.core as mx

from benches import (
    bench,
    bench_course_progression,
    profile_week2_kernels,
    week2_gpudebug,
)
from benches.bench_course_progression import WEEK2_VARIANTS
from benches.profile_week2_kernels import DEFAULT_CASES
from benches.week2_gpudebug import KNOWN_CHECKPOINTS
from .tiny_llm_base import Qwen3ModelWeek2, quantized_matmul, quantized_matmul_vanilla
from .utils import assert_allclose, tiny_qwen3_mlx_model


def test_task_1_simd_matmul_checkpoint_runs_the_week2_engine():
    fixture = tiny_qwen3_mlx_model()
    tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
    with mx.stream(mx.gpu):
        simd_model = Qwen3ModelWeek2(fixture, checkpoint="simd-matmul")
        readable_model = Qwen3ModelWeek2(fixture, checkpoint="quantized-matvec")
        simd_output = simd_model(tokens, 0, simd_model.create_kv_cache(capacity=3))
        readable_output = readable_model(
            tokens, 0, readable_model.create_kv_cache(capacity=3)
        )
        mx.eval(simd_output, readable_output)

    assert simd_output.dtype == readable_output.dtype == mx.bfloat16
    assert simd_output.shape == readable_output.shape
    assert simd_output.shape[:2] == (1, 3)
    assert_allclose(simd_output, readable_output, mx.bfloat16, atol=0.25, rtol=1e-2)


def test_task_2_simdgroup_matmul_matches_readable_partial_tiles_gpu():
    with mx.stream(mx.gpu):
        inputs = mx.random.normal((10, 256)).astype(mx.bfloat16)
        weight = mx.random.normal((97, 256)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        tiled = quantized_matmul(
            scales,
            biases,
            128,
            4,
            inputs,
            packed,
            transpose_b=True,
            use_simdgroup=True,
        )
        readable = quantized_matmul_vanilla(
            scales, biases, 128, 4, inputs, packed, transpose_b=True
        )
        assert tiled is not None, "implement the SIMD quantized_matmul learner seam"
        assert_allclose(tiled, readable, mx.bfloat16, atol=0.25, rtol=1e-2)


def test_task_3_simd_matmul_model_prefill_matches_readable_control_gpu():
    fixture = tiny_qwen3_mlx_model()
    tokens = mx.array([list(range(1, 11))], dtype=mx.int32)
    with mx.stream(mx.gpu):
        simd_model = Qwen3ModelWeek2(fixture, checkpoint="simd-matmul")
        readable_model = Qwen3ModelWeek2(fixture, checkpoint="quantized-matvec")
        simd_output = simd_model(tokens, 0, simd_model.create_kv_cache(capacity=10))
        readable_output = readable_model(
            tokens, 0, readable_model.create_kv_cache(capacity=10)
        )
        mx.eval(simd_output, readable_output)

    assert simd_output.dtype == readable_output.dtype == mx.bfloat16
    assert simd_output.shape == readable_output.shape
    assert_allclose(simd_output, readable_output, mx.bfloat16, atol=0.5, rtol=5e-2)


def test_day3_public_selectors_stop_at_simd_matmul():
    model_module = importlib.import_module(Qwen3ModelWeek2.__module__)
    checkpoints = (
        "kv-cache",
        "capacity-cache",
        "quantized-matvec",
        "simd-matmul",
    )
    assert model_module.WEEK2_CHECKPOINTS == checkpoints
    assert KNOWN_CHECKPOINTS == checkpoints
    assert DEFAULT_CASES == (
        "kv-cache:decode:128",
        "capacity-cache:decode:128",
        "quantized-matvec:decode:128",
        "simd-matmul:prefill:128",
    )
    assert [variant.key for variant in WEEK2_VARIANTS] == [
        "week1",
        "week2-kv-cache",
        "week2-capacity-cache",
        "week2-quantized-matvec",
        "week2-simd-matmul",
        "mlx",
    ]


def test_day3_model_free_cli_parsers(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--solution",
            "tiny_llm_ref",
            "--loader",
            "week2",
            "--week2-checkpoint",
            "simd-matmul",
            "--model",
            "qwen3-0.6b",
            "--prefill-logits",
            "last",
        ],
    )
    bench_args = bench.parse_args()
    bench.validate_args(bench_args)
    assert bench_args.week2_checkpoint == "simd-matmul"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "progression",
            "--solution",
            "ref",
            "--suite",
            "week2",
            "--repeats",
            "2",
            "--variant",
            "week2-quantized-matvec",
            "--variant",
            "week2-simd-matmul",
            "--prefill-logits",
            "last",
        ],
    )
    assert bench_course_progression.parse_args().variant == [
        "week2-quantized-matvec",
        "week2-simd-matmul",
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--solution",
            "tiny_llm_ref",
            "--case",
            "simd-matmul:prefill:128",
        ],
    )
    assert profile_week2_kernels.parse_args().case == [
        profile_week2_kernels.ProfileCase("simd-matmul", "prefill", 128)
    ]

    capture = week2_gpudebug.build_parser().parse_args(
        [
            "capture",
            "--solution",
            "tiny_llm_ref",
            "--model",
            "qwen3-0.6b",
            "--checkpoint",
            "simd-matmul",
            "--phase",
            "prefill",
            "--tokens",
            "128",
            "--trace",
            "trace.gputrace",
            "--metadata",
            "metadata.json",
            "--manifest",
            "manifest.json",
        ]
    )
    assert capture.checkpoint == "simd-matmul"
