import json
import re
import unicodedata
from pathlib import Path

import pytest

from benches.bench_course_progression import WEEK2_VARIANTS


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_PROFILING_LABEL = "optional profiling evidence"
APPROVED_OPTIONAL_EVIDENCE = frozenset(
    {
        "optional profiling evidence a kernel group replay or operator attribution can "
        "corroborate that transition but neither gates progress the reference checkpoint "
        "includes both alongside the model and projection measurements above",
        "optional profiling evidence the day 3 kernel group replay and the reference "
        "solution attribution show the pointwise cluster behind the optimized projections "
        "they explain the chapter order but are not prerequisites or acceptance gates",
        "optional profiling evidence the reference checkpoint pairs the cumulative and "
        "operator measurements with an updated attribution that attribution can explain the "
        "transition but it does not replace the checkpoint evidence above",
        "optional profiling evidence decode and prefill kernel group results can explain how "
        "the workload divides its time but they are reference evidence not required output "
        "for this checkpoint",
        "optional profiling evidence the reference checkpoint pairs the context sweep short "
        "context model delta and fixed workload control with a separate prefill attribution "
        "the attribution explains why the course targets matrix shaped projections next it "
        "is not a prerequisite for day 6",
        "optional profiling evidence the checked dependency aware attribution and the "
        "reference solution attribution explain why projections are the reference solution s "
        "next target they are not required learner output and do not gate this chapter",
        "optional profiling evidence a 32 128 row attribution can corroborate the shape "
        "analysis but it does not replace the matched complete model delta projection "
        "controls and dispatch calculation above",
    }
)
APPROVED_REQUIRED_TRACE = re.compile(
    r"\b(?:direct )?(?:fused dispatch )?(?:source|dispatch) traces?\b",
    re.IGNORECASE,
)
PROFILING_ONLY = re.compile(
    r"\bprofil(?:e|ed|es|er|ers|ing)\b|"
    r"\battribut(?:e|ed|es|ing|ion|ions)\b|"
    r"\bkernel groups?\b|"
    r"\boperator breakdowns?\b|"
    r"\breplay(?:ed|s|ing)?\b|"
    r"\bxcode\b|"
    r"\bcaptur(?:e|ed|es|ing)\b|"
    r"\bgpudebug\b|"
    r"\bgputrace\b|"
    r"\btimelines?\b|"
    r"\bmetal system trace\b|"
    r"\bscreenshots?\b|"
    r"\bgpu durations?\b|"
    r"\btrac(?:e|ed|es|ing)\b",
    re.IGNORECASE,
)


def _normalize_contract_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = re.sub(r"[*_`~]+", " ", text)
    text = re.sub(r"[-‐‑‒–—_/]+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _assert_required_progression_is_profile_free(chapter: str, day: int) -> None:
    optional_blocks = 0

    for paragraph in re.split(r"\n\s*\n", chapter):
        normalized = _normalize_contract_text(paragraph)
        if normalized.startswith(OPTIONAL_PROFILING_LABEL):
            optional_blocks += 1
            assert normalized in APPROVED_OPTIONAL_EVIDENCE, (
                f"Day {day} optional profiling block contains required semantics or is not "
                f"an approved evidence-only contract: {normalized!r}"
            )
            continue

        required_text = APPROVED_REQUIRED_TRACE.sub("", normalized)
        profiling_match = PROFILING_ONLY.search(required_text)
        assert profiling_match is None, (
            f"Day {day} makes profiling output part of required progression: "
            f"{profiling_match.group(0)!r}"
        )

    assert optional_blocks, f"Day {day} must label its optional evidence"


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


def test_required_week2_progression_never_depends_on_optional_profiling():
    days = {
        day: (ROOT / "book/src" / filename).read_text()
        for day, filename in {
            3: "week2-03-quantize-model.md",
            4: "week2-04-fused-model-kernels.md",
            5: "week2-05-decode-attention.md",
            6: "week2-06-simd-matrix-prefill.md",
        }.items()
    }

    for day, chapter in days.items():
        _assert_required_progression_is_profile_free(chapter, day)

    day3 = days[3]
    assert "matrix schedule. Use a source trace through those branches" in day3
    assert "direct source trace of the dispatch branches" in day3
    assert "source trace proves" in day3


@pytest.mark.parametrize(
    "required_mutation",
    (
        "Attach the Xcode GPU capture before continuing.",
        "Record a gpudebug timeline as the acceptance gate.",
        "Require the Metal System Trace and screenshot before Day 5.",
        "Continue only when cumulative GPU duration shrinks.",
        "Attach the GPU-duration result before Day 5.",
        "Attach the kernel **group** evidence before Day 5.",
        "The operator breakdown must be attached before Day 5.",
        "Complete the checkpoint by attaching the operator breakdown.",
        "Progress requires the operator breakdown attachment.",
        "Attach the .gputrace before continuing.",
        "Record a trace before continuing.",
    ),
)
def test_required_profiling_vocabulary_mutations_fail_closed(required_mutation):
    chapter = (
        "> **Optional profiling evidence.** A kernel-group replay or operator attribution "
        "can corroborate that transition, but neither gates progress. The "
        "[reference checkpoint](./appendix-performance.md#day-3-keep-weights-packed) "
        "includes both alongside the model and projection measurements above.\n\n"
        f"{required_mutation}"
    )

    with pytest.raises(AssertionError, match="profiling output part"):
        _assert_required_progression_is_profile_free(chapter, 4)


@pytest.mark.parametrize(
    "optional_mutation",
    (
        "The replay is required.",
        "The replay is required before the learner may continue.",
        "You must attach the attribution.",
        "Do not continue until the kernel-group replay is available.",
        "The attribution is a prerequisite for Day 5.",
        "The replay is a condition for advancing to Day 5.",
        "Proceed only after attaching the attribution.",
        "The screenshot is mandatory.",
        "The replay is an acceptance criterion.",
        "You need the attribution to continue.",
        "The replay is necessary to advance to Day 5.",
        "The screenshot is essential for Day 5.",
        "Day 5 depends on attaching the attribution.",
        "Only after the replay may you continue.",
        "Day 5 starts only after the replay is attached.",
        "Only learners with the replay may proceed.",
        "Complete this checkpoint by attaching the operator breakdown.",
        "Progress depends on the screenshot attachment.",
    ),
)
def test_optional_profiling_gate_mutations_fail_closed(optional_mutation):
    chapter = (
        "The direct dispatch trace reaches the intended kernel.\n\n"
        f"> **Optional profiling evidence.** {optional_mutation}"
    )

    with pytest.raises(AssertionError, match="required semantics"):
        _assert_required_progression_is_profile_free(chapter, 4)


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
