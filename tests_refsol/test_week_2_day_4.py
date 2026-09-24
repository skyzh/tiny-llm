"""Week 2 Day 4 compact model-primitive tests."""

import importlib
import sys

import mlx.core as mx
import pytest

from benches import (
    bench,
    bench_course_progression,
    profile_week2_kernels,
    week2_gpudebug,
)
from benches.bench_course_progression import WEEK2_VARIANTS
from benches.profile_week2_kernels import DEFAULT_CASES
from benches.week2_gpudebug import KNOWN_CHECKPOINTS
from .tiny_llm_base import FastRMSNorm, FastRoPE, Qwen3ModelWeek2, swiglu
from .utils import assert_allclose, tiny_qwen3_mlx_model


def test_task_1_register_cached_rmsnorm_matches_readable_operator():
    x = mx.random.normal((2, 3, 16)).astype(mx.bfloat16)
    weight = mx.random.normal((16,)).astype(mx.bfloat16)
    result = FastRMSNorm(16, weight, eps=1e-5)(x)
    expected = mx.fast.rms_norm(x, weight, 1e-5)

    assert result is not None, "implement the FastRMSNorm learner seam"
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype == mx.bfloat16
    assert_allclose(result, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


def test_task_2_rope_matches_readable_operator():
    x = mx.random.normal((2, 4, 2, 16)).astype(mx.bfloat16)
    actual = FastRoPE(16, 32, base=10000)(x, [3, 7])
    expected = mx.fast.rope(
        x.transpose(0, 2, 1, 3),
        16,
        traditional=False,
        base=10000,
        scale=1.0,
        offset=mx.array([3, 7], dtype=mx.int32),
    ).transpose(0, 2, 1, 3)

    assert actual is not None, "implement the FastRoPE learner seam"
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == mx.bfloat16
    assert_allclose(actual, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


def test_task_3_swiglu_matches_readable_operator():
    gate = mx.random.normal((2, 4, 16)).astype(mx.bfloat16)
    up = mx.random.normal((2, 4, 16)).astype(mx.bfloat16)
    actual = swiglu(gate, up)
    expected = gate * mx.sigmoid(gate) * up

    assert actual is not None, "implement the SwiGLU learner seam"
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == mx.bfloat16
    assert_allclose(actual, expected, mx.bfloat16, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("seed", (0, 4))
def test_task_4_primitive_checkpoints_compose_public_model(seed):
    random_state = mx.random.state[:]
    try:
        mx.random.seed(seed)
        fixture = tiny_qwen3_mlx_model()
    finally:
        mx.random.state[:] = random_state

    tokens = mx.array([list(range(1, 11))], dtype=mx.int32)
    outputs = []
    for checkpoint in ("simd-matmul", "rmsnorm", "rope", "swiglu"):
        model = Qwen3ModelWeek2(fixture, checkpoint=checkpoint)
        output = model(tokens, 0, model.create_kv_cache(capacity=10))
        mx.eval(output)
        assert output.dtype == mx.bfloat16
        assert output.shape[:2] == (1, 10)
        outputs.append(output)

    assert all(output.shape == outputs[0].shape for output in outputs)
    for earlier, later in zip(outputs, outputs[1:]):
        assert_allclose(
            later,
            earlier,
            mx.bfloat16,
            atol=0.5,
            rtol=2e-2,
            message=f"fixture seed {seed}",
        )


def test_day4_public_selectors_include_only_shipped_checkpoints():
    model_module = importlib.import_module(Qwen3ModelWeek2.__module__)
    checkpoints = (
        "kv-cache",
        "capacity-cache",
        "quantized-matvec",
        "simd-matmul",
        "rmsnorm",
        "rope",
        "swiglu",
    )
    assert model_module.WEEK2_CHECKPOINTS == checkpoints
    assert KNOWN_CHECKPOINTS == checkpoints
    assert DEFAULT_CASES == (
        "kv-cache:decode:128",
        "capacity-cache:decode:128",
        "quantized-matvec:decode:128",
        "simd-matmul:prefill:128",
        "rmsnorm:decode:128",
        "rope:decode:128",
        "swiglu:decode:128",
    )
    assert [variant.key for variant in WEEK2_VARIANTS] == [
        "week1",
        "week2-kv-cache",
        "week2-capacity-cache",
        "week2-quantized-matvec",
        "week2-simd-matmul",
        "week2-rmsnorm",
        "week2-rope",
        "week2-swiglu",
        "mlx",
    ]


def test_day4_model_free_cli_parsers(monkeypatch):
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
            "swiglu",
            "--model",
            "qwen3-0.6b",
            "--prefill-logits",
            "last",
        ],
    )
    bench_args = bench.parse_args()
    bench.validate_args(bench_args)
    assert bench_args.week2_checkpoint == "swiglu"

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
            "week2-rmsnorm",
            "--variant",
            "week2-rope",
            "--variant",
            "week2-swiglu",
            "--prefill-logits",
            "last",
        ],
    )
    assert bench_course_progression.parse_args().variant == [
        "week2-rmsnorm",
        "week2-rope",
        "week2-swiglu",
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        ["profile", "--solution", "tiny_llm_ref", "--case", "swiglu:decode:128"],
    )
    assert profile_week2_kernels.parse_args().case == [
        profile_week2_kernels.ProfileCase("swiglu", "decode", 128)
    ]

    capture = week2_gpudebug.build_parser().parse_args(
        [
            "capture",
            "--solution",
            "tiny_llm_ref",
            "--model",
            "qwen3-0.6b",
            "--checkpoint",
            "swiglu",
            "--phase",
            "decode",
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
    assert capture.checkpoint == "swiglu"
