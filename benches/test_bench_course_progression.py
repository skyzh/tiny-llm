import json
import re
from pathlib import Path

import pytest

from benches.bench_course_progression import WEEK2_VARIANTS


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_PROFILING_PREFIX = "> **Optional profiling evidence.**"
APPROVED_REQUIRED_TRACE = re.compile(
    r"\b(?:direct\s+)?(?:fused[- ]dispatch\s+)?"
    r"(?:source|dispatch)\s+traces?\b",
    re.IGNORECASE,
)
PROFILING_ONLY = re.compile(
    r"\bprofil(?:e|ed|es|er|ers|ing)\b|"
    r"\battribut(?:e|ed|es|ing|ion|ions)\b|"
    r"\bkernel[- ]groups?\b|"
    r"\breplay(?:ed|s|ing)?\b|"
    r"\bxcode\b|"
    r"\bcaptur(?:e|ed|es|ing)\b|"
    r"\bgpudebug\b|"
    r"(?:\.gputrace|\bgputrace\b)|"
    r"\btimelines?\b|"
    r"\bmetal system trace\b|"
    r"\bscreenshots?\b|"
    r"\bgpu durations?\b|"
    r"\btrac(?:e|ed|es|ing)\b",
    re.IGNORECASE,
)
NEGATED_OPTIONAL_GATE = re.compile(
    r"\bnot required\b|"
    r"\bdo(?:es)? not require\b|"
    r"\bneither gates? progress\b|"
    r"\bdo(?:es)? not gate(?: progress| this chapter| the checkpoint)?\b|"
    r"\bnot prerequisites? or acceptance gates?\b|"
    r"\bnot (?:a |an )?prerequisites?\b|"
    r"\bnot (?:a |an )?acceptance gates?\b",
    re.IGNORECASE,
)
OPTIONAL_GATE_SEMANTICS = re.compile(
    r"\brequir(?:e|es|ed|ing)\b|"
    r"\bmust\b|"
    r"\bmandatory\b|"
    r"\bbefore (?:the learner (?:may|can) )?"
    r"(?:continue|continuing|proceeding|advancing|moving on|day \d+)\b|"
    r"\b(?:do not|don't) continue\b|"
    r"\b(?:cannot|can't) continue\b|"
    r"\b(?:continue|proceed|advance) (?:only|after|when|once)\b|"
    r"\bprerequisites?\b|"
    r"\bgates?\b|"
    r"\bblocks? progress\b|"
    r"\bacceptance criteri(?:on|a)\b|"
    r"\bprogression requirements?\b|"
    r"\b(?:condition|criterion) (?:to|for) "
    r"(?:continue|continuing|proceed|proceeding|advance|advancing)\b|"
    r"\bincomplete until\b",
    re.IGNORECASE,
)


def _assert_required_progression_is_profile_free(chapter: str, day: int) -> None:
    optional_blocks = 0

    for paragraph in re.split(r"\n\s*\n", chapter):
        normalized = re.sub(r"\s+", " ", paragraph.replace("\n> ", "\n"))
        if paragraph.lstrip().startswith(OPTIONAL_PROFILING_PREFIX):
            optional_blocks += 1
            gate_text = NEGATED_OPTIONAL_GATE.sub("", normalized)
            gate_match = OPTIONAL_GATE_SEMANTICS.search(gate_text)
            assert gate_match is None, (
                f"Day {day} optional profiling block contains required semantics: "
                f"{gate_match.group(0)!r}"
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
        "Attach the .gputrace before continuing.",
        "Record a trace before continuing.",
    ),
)
def test_required_profiling_vocabulary_mutations_fail_closed(required_mutation):
    chapter = (
        "> **Optional profiling evidence.** A replay can explain the result, "
        "but it does not replace the checkpoint evidence.\n\n"
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
