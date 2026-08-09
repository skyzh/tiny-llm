import json
from pathlib import Path

from benches.bench_course_progression import WEEK2_VARIANTS


ROOT = Path(__file__).resolve().parents[1]


def test_week2_live_labels_follow_the_six_day_book():
    labels = {variant.key: variant.label for variant in WEEK2_VARIANTS}
    assert labels == {
        "week1": "Week 1 readable",
        "week2-kv-cache": "2.1 KV cache",
        "week2-quantized-matvec": "2.2 Quantized matvec",
        "week2-rmsnorm": "2.3 Fast RMSNorm",
        "week2-rope": "2.3 + Fast RoPE",
        "week2-swiglu": "2.3 + Fused SwiGLU",
        "week2-decode-attention": "2.4 Decode attention",
        "week2-simd-matmul": "2.5 SIMD matrix prefill",
        "week2-split-k": "2.6 Split-K prefill",
        "mlx": "MLX",
    }

    readme = (ROOT / "README.md").read_text()
    summary = (ROOT / "book/src/SUMMARY.md").read_text()
    assert "| 2.2 | Benchmark, Profile, and Quantize |" in readme
    assert "| 2.3 | Fused Model Kernels |" in readme
    assert "| 2.4 | Fused Decode Attention |" in readme
    assert "| 2.5 | SIMD-Matrix Prefill |" in readme
    assert "| 2.6 | Split-K Prefill |" in readme
    assert "./week2-02-benchmark-quantize.md" in summary
    assert "./week2-03-fused-model-kernels.md" in summary
    assert "./week2-04-decode-attention.md" in summary
    assert "./week2-05-simd-matrix-prefill.md" in summary
    assert "./week2-06-split-k-prefill.md" in summary

    chapter_headings = {
        "week2-02-benchmark-quantize.md": (
            "# 🚧 Week 2 Day 2: Benchmark, Profile, and Quantize"
        ),
        "week2-03-fused-model-kernels.md": "# 🚧 Week 2 Day 3: Fused Model Kernels",
        "week2-04-decode-attention.md": "# 🚧 Week 2 Day 4: Fused Decode Attention",
        "week2-05-simd-matrix-prefill.md": "# 🚧 Week 2 Day 5: SIMD-Matrix Prefill",
        "week2-06-split-k-prefill.md": "# 🚧 Week 2 Day 6: Split-K Prefill",
    }
    for filename, expected_heading in chapter_headings.items():
        heading = (ROOT / "book/src" / filename).read_text().splitlines()[0]
        assert heading == expected_heading


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
