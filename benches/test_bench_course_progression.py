import json
from pathlib import Path

from benches.bench_course_progression import WEEK2_VARIANTS


ROOT = Path(__file__).resolve().parents[1]


def test_week2_live_labels_follow_the_seven_day_book():
    labels = {variant.key: variant.label for variant in WEEK2_VARIANTS}
    assert labels == {
        "week1": "Week 1 readable",
        "week2-kv-cache": "2.1 KV cache",
        "week2-quantized-matvec": "2.3 Quantized matvec",
        "week2-rmsnorm": "2.4 Fast RMSNorm",
        "week2-rope": "2.4 + Fast RoPE",
        "week2-swiglu": "2.4 + Fused SwiGLU",
        "week2-decode-attention": "2.5 Decode attention",
        "week2-simd-matmul": "2.6 SIMD matrix prefill",
        "week2-split-k": "2.7 Split-K prefill",
        "mlx": "MLX",
    }

    readme = (ROOT / "README.md").read_text()
    summary = (ROOT / "book/src/SUMMARY.md").read_text()
    assert "| 2.2 | Benchmarking and Profiling |" in readme
    assert "| 2.3 | Quantize the Model |" in readme
    assert "| 2.4 | Fused Model Kernels |" in readme
    assert "| 2.5 | Fused Decode Attention |" in readme
    assert "| 2.6 | SIMD-Matrix Prefill |" in readme
    assert "| 2.7 | Split-K Prefill |" in readme
    assert "./week2-02-benchmark-profile.md" in summary
    assert "./week2-03-quantize-model.md" in summary
    assert "./week2-04-fused-model-kernels.md" in summary
    assert "./week2-05-decode-attention.md" in summary
    assert "./week2-06-simd-matrix-prefill.md" in summary
    assert "./week2-07-split-k-prefill.md" in summary

    chapter_headings = {
        "week2-02-benchmark-profile.md": (
            "# 🚧 Week 2 Day 2: Benchmarking and Profiling"
        ),
        "week2-03-quantize-model.md": "# 🚧 Week 2 Day 3: Quantize the Model",
        "week2-04-fused-model-kernels.md": "# 🚧 Week 2 Day 4: Fused Model Kernels",
        "week2-05-decode-attention.md": "# 🚧 Week 2 Day 5: Fused Decode Attention",
        "week2-06-simd-matrix-prefill.md": "# 🚧 Week 2 Day 6: SIMD-Matrix Prefill",
        "week2-07-split-k-prefill.md": "# 🚧 Week 2 Day 7: Split-K Prefill",
    }
    for filename, expected_heading in chapter_headings.items():
        heading = (ROOT / "book/src" / filename).read_text().splitlines()[0]
        assert heading == expected_heading


def test_week2_profile_boundary_is_optional_and_quantization_is_day_3():
    day2 = (ROOT / "book/src/week2-02-benchmark-profile.md").read_text()
    day3 = (ROOT / "book/src/week2-03-quantize-model.md").read_text()
    appendix = (ROOT / "book/src/week2-advanced-profiling.md").read_text()

    assert "pdm run bench" in day2
    assert "Profiling is optional" in day2
    assert "macOS 27" in day2
    assert "pdm run test --week 2 --day 3" in day3
    assert "Week 2 Day 2: Benchmark, Profile, and Quantize" not in day3

    removed_workflow_tokens = (
        "capture-week2-shader",
        "MLX_METAL_DEBUG",
        "MTL_CAPTURE_ENABLED",
        ".gputrace",
        "profile-week2-kernels",
    )
    live_week2 = "\n".join(
        (ROOT / "book/src" / f"week2-0{day}-{name}.md").read_text()
        for day, name in (
            (2, "benchmark-profile"),
            (3, "quantize-model"),
            (4, "fused-model-kernels"),
            (5, "decode-attention"),
            (6, "simd-matrix-prefill"),
            (7, "split-k-prefill"),
        )
    )
    assert not any(token in live_week2 for token in removed_workflow_tokens)
    assert not (ROOT / "benches/capture_week2_shader.py").exists()
    assert not (ROOT / "benches/test_capture_week2_shader.py").exists()
    assert "capture-week2-shader" not in (ROOT / "pyproject.toml").read_text()
    assert "not required" in appendix
    assert "macOS 27" in appendix


def test_historical_week2_artifact_keeps_original_labels():
    artifact = json.loads(
        (
            ROOT / "benchmark_results/m4-pro-qwen3-4b-week2-progression-mlx-0.32.0.json"
        ).read_text()
    )
    labels = {key: result["label"] for key, result in artifact["results"].items()}
    assert labels == {
        "week2-kv-cache": "2.1 KV cache",
        "week2-quantized-matvec": "2.3 Quantized matvec",
        "week2-rmsnorm": "2.4 Fast RMSNorm",
        "week2-rope": "2.4 + Fast RoPE",
        "week2-swiglu": "2.4 + Fused SwiGLU",
        "week2-decode-attention": "2.5 Decode attention",
        "week2-simd-matmul": "2.6 SIMD matrix prefill",
        "week2-split-k": "2.7 Split-K prefill",
        "mlx": "MLX",
    }
