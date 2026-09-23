import json
import re
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

from benches import bench as benchmark
from benches import bench_course_progression as progression


WEEK2_VARIANTS = progression.WEEK2_VARIANTS


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


def test_week2_live_labels_follow_the_five_day_book():
    labels = {variant.key: variant.label for variant in WEEK2_VARIANTS}
    assert labels == {
        "week1": "Week 1 readable",
        "week2-kv-cache": "2.1 Reuse the prefix",
        "week2-capacity-cache": "2.1 + Bound KV-cache movement",
        "week2-quantized-matvec": "2.2 Keep W4 packed",
        "week2-simd-matmul": "2.3 SIMD matrix prefill",
        "week2-rmsnorm": "2.4 Compact RMSNorm",
        "week2-rope": "2.4 + Compact RoPE",
        "week2-swiglu": "2.4 + Compact SwiGLU",
        "week2-tiled-prefill": "2.5 Tiled dense prefill attention",
        "week2-selected": "2.5 + Run the selected inference engine",
        "mlx": "MLX",
    }

    summary = (ROOT / "book/src/SUMMARY.md").read_text()
    week2_summary = summary.split("Week 2:", 1)[1].split("Week 3:", 1)[0]
    for title in (
        "Day 1: Cache and Measure",
        "Day 2: Keep W4 Packed",
        "Day 3: SIMD Matrix Prefill",
        "Day 4: Fused Model Primitives",
        "Day 5: Tiled Dense Prefill Attention",
    ):
        assert title in week2_summary
    assert len(re.findall(r"\[🚧 Day \d:", week2_summary)) == 5

    chapter_headings = {
        "week2-01-kv-cache.md": "# 🚧 Week 2 Day 1: Reuse the Prefix, Then Bound the Cache",
        "week2-02-quantize-model.md": "# 🚧 Week 2 Day 2: Keep W4 Packed",
        "week2-03-simd-matrix-prefill.md": "# 🚧 Week 2 Day 3: SIMD-Matrix Prefill",
        "week2-04-fused-model-kernels.md": "# 🚧 Week 2 Day 4: Fused Model Kernels",
        "week2-05-tiled-prefill-attention.md": "# 🚧 Week 2 Day 5: Tiled Dense Prefill Attention",
    }
    for filename, expected_heading in chapter_headings.items():
        heading = (ROOT / "book/src" / filename).read_text().splitlines()[0]
        assert heading == expected_heading

    readme = (ROOT / "README.md").read_text()
    week2_overview = readme.split("- **Week 2:", 1)[1].split("- **Week 3:", 1)[0]
    assert "decode attention" not in week2_overview.lower()
    assert "split-k" not in week2_overview.lower()
    for row in (
        "| 2.1 | Cache and Measure |",
        "| 2.2 | Keep W4 Packed |",
        "| 2.3 | SIMD Matrix Prefill |",
        "| 2.4 | Fused Model Primitives |",
        "| 2.5 | Tiled Dense Prefill Attention |",
    ):
        assert row in readme
    assert "| 2.6 |" not in readme
    assert "| 2.7 |" not in readme


def test_week2_progression_checkpoints_parse_through_public_bench(monkeypatch):
    checkpoints = tuple(
        variant.extra_args[1]
        for variant in WEEK2_VARIANTS
        if variant.loader == "week2" and variant.extra_args
    )
    assert checkpoints == (
        "kv-cache",
        "capacity-cache",
        "quantized-matvec",
        "simd-matmul",
        "rmsnorm",
        "rope",
        "swiglu",
        "tiled-prefill",
        "selected",
    )

    monkeypatch.setattr(
        benchmark,
        "load",
        lambda *_args, **_kwargs: pytest.fail("model work must not begin"),
    )
    for checkpoint in checkpoints:
        monkeypatch.setattr(
            sys,
            "argv",
            ["bench", "--loader", "week2", "--week2-checkpoint", checkpoint],
        )
        assert benchmark.parse_args().week2_checkpoint == checkpoint

    for retired in ("decode-attention", "split-k"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["bench", "--loader", "week2", "--week2-checkpoint", retired],
        )
        with pytest.raises(SystemExit):
            benchmark.parse_args()


def test_benchmark_refuses_existing_json_before_host_or_model_work(
    tmp_path, monkeypatch
):
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    monkeypatch.setattr(
        progression,
        "parse_args",
        lambda: SimpleNamespace(json_output=output),
    )
    monkeypatch.setattr(
        progression,
        "collect_host_metadata",
        lambda: pytest.fail("host/model work must not begin"),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        progression.main()


def test_week2_profile_boundary_is_optional_and_quantization_is_day_2():
    day1 = (ROOT / "book/src/week2-01-kv-cache.md").read_text()
    day2 = (ROOT / "book/src/week2-02-quantize-model.md").read_text()
    optional_capture = (ROOT / "book/src/week2-advanced-profiling.md").read_text()
    overview = (ROOT / "book/src/week2-overview.md").read_text()

    assert "--week2-checkpoint kv-cache" in day1
    assert "--week2-checkpoint capacity-cache" in day1
    assert "pdm run test --week 2 --day 1" in day1
    assert "pdm run test --week 2 --day 2" in day2
    assert "--week2-checkpoint quantized-matvec" in day2
    assert "[optional macOS capture lab]" in overview
    assert "It is never an acceptance gate." in optional_capture
    assert "pdm run test --week 2 --day 4 -- -k swiglu" in optional_capture
    scripts = (ROOT / "pyproject.toml").read_text()
    assert "capture-week2" in scripts
    assert "reduce-week2-gpudebug" in scripts


def test_required_week2_progression_uses_portable_checks_not_local_capture():
    days = {
        day: (ROOT / "book/src" / filename).read_text()
        for day, filename in {
            1: "week2-01-kv-cache.md",
            2: "week2-02-quantize-model.md",
            3: "week2-03-simd-matrix-prefill.md",
            4: "week2-04-fused-model-kernels.md",
            5: "week2-05-tiled-prefill-attention.md",
        }.items()
    }
    expected_checkpoints = {
        1: ("kv-cache", "capacity-cache"),
        2: ("quantized-matvec",),
        3: ("simd-matmul",),
        4: ("rmsnorm", "rope", "swiglu"),
        5: ("tiled-prefill", "selected"),
    }
    for day, chapter in days.items():
        assert all(checkpoint in chapter for checkpoint in expected_checkpoints[day])
        assert f"pdm run test --week 2 --day {day}" in chapter
        assert "--week2-checkpoint decode-attention" not in chapter
        assert "--week2-checkpoint split-k" not in chapter
    assert "`gpudebug` output, screenshot, or device-specific counter gates" in days[1]
    for day in (2, 3, 4, 5):
        assert not any(
            token in days[day]
            for token in (
                "Xcode GPU capture",
                "Metal System Trace",
                ".gputrace",
                "gpudebug",
                "screenshot",
                "GPU duration",
            )
        )


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
