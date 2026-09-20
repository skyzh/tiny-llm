import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

from benches import bench
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
        "optional profiling evidence a 128 8k component replay can corroborate the "
        "attention crossover but it does not replace the matched product matrix "
        "selection counter or disable only control above",
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
        "week2-simd-matmul": "2.5 SIMD matrix prefill",
        "week2-shared-input-qkv": "2.6 Shared-input QKV",
        "week2-shared-input-gate-up-swiglu": "2.6 Shared-input gate+up/SwiGLU",
        "week2-io-aware-dense-attention": "2.7 I/O-aware dense attention",
        "mlx": "MLX",
    }

    readme = (ROOT / "README.md").read_text()
    summary = (ROOT / "book/src/SUMMARY.md").read_text()
    assert "| 2.2 | Benchmarking and Profiling |" in readme
    assert "| 2.3 | Quantize the Model |" in readme
    assert "| 2.4 | Fused Model Kernels |" in readme
    assert "| 2.5 | SIMD-Matrix Prefill |" in readme
    assert "| 2.6 (optional) | Workload-Conditioned Operator Lab |" in readme
    assert "| 2.7 | Conditional Split-K and Final Decision |" in readme
    assert "./week2-02-benchmark-profile.md" in summary
    assert "./week2-03-quantize-model.md" in summary
    assert "./week2-04-fused-model-kernels.md" in summary
    assert "./week2-05-simd-matrix-prefill.md" in summary
    assert "./week2-06-operator-lab.md" in summary
    assert "./week2-07-split-k-prefill.md" in summary

    chapter_headings = {
        "week2-02-benchmark-profile.md": (
            "# 🚧 Week 2 Day 2: Benchmarking and Profiling"
        ),
        "week2-03-quantize-model.md": "# 🚧 Week 2 Day 3: Quantize the Model",
        "week2-04-fused-model-kernels.md": "# 🚧 Week 2 Day 4: Fused Model Kernels",
        "week2-05-simd-matrix-prefill.md": "# 🚧 Week 2 Day 5: SIMD-Matrix Prefill",
        "week2-06-operator-lab.md": (
            "# 🚧 Week 2 Day 6 (Optional): Workload-Conditioned Operator Lab"
        ),
        "week2-07-split-k-prefill.md": (
            "# 🚧 Week 2 Day 7: Conditional Split-K and Final Decision"
        ),
    }
    for filename, expected_heading in chapter_headings.items():
        heading = (ROOT / "book/src" / filename).read_text().splitlines()[0]
        assert heading == expected_heading


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


def test_week2_profile_boundary_is_optional_and_quantization_is_day_3():
    day2 = (ROOT / "book/src/week2-02-benchmark-profile.md").read_text()
    day3 = (ROOT / "book/src/week2-03-quantize-model.md").read_text()
    appendix = (ROOT / "book/src/week2-advanced-profiling.md").read_text()

    assert "pdm run bench" in day2
    assert "capture remains optional" in day2
    assert "pdm run profile-week2-kernels --solution tiny_llm" in day2
    assert "macOS 27" in day2
    assert "pdm run test --week 2 --day 3" in day3
    assert "Week 2 Day 2: Benchmark, Profile, and Quantize" not in day3

    removed_workflow_tokens = ("capture-week2-shader", "MLX_METAL_DEBUG")
    live_week2 = "\n".join(
        (ROOT / "book/src" / f"week2-0{day}-{name}.md").read_text()
        for day, name in (
            (2, "benchmark-profile"),
            (3, "quantize-model"),
            (4, "fused-model-kernels"),
            (5, "simd-matrix-prefill"),
            (6, "operator-lab"),
            (7, "split-k-prefill"),
        )
    )
    assert not any(token in live_week2 for token in removed_workflow_tokens)
    scripts = (ROOT / "pyproject.toml").read_text()
    assert "capture-week2" in scripts
    assert "reduce-week2-gpudebug" in scripts
    assert "never an acceptance gate" in appendix
    assert "macOS 27" in appendix


def test_required_week2_progression_uses_portable_attribution_not_local_capture():
    days = {
        day: (ROOT / "book/src" / filename).read_text()
        for day, filename in {
            3: "week2-03-quantize-model.md",
            4: "week2-04-fused-model-kernels.md",
            5: "week2-05-simd-matrix-prefill.md",
        }.items()
    }

    expected_cases = {
        3: ("kv-cache:decode:128", "quantized-matvec:decode:128"),
        4: ("quantized-matvec:decode:128", "swiglu:decode:128"),
        5: ("swiglu:prefill:128", "simd-matmul:prefill:128"),
    }
    local_capture_tokens = (
        "Xcode GPU capture",
        "Metal System Trace",
        ".gputrace",
        "gpudebug",
        "screenshot",
        "GPU duration",
    )

    for day, chapter in days.items():
        assert "pdm run profile-week2-kernels --solution tiny_llm" in chapter
        assert all(case in chapter for case in expected_cases[day])
        assert not any(token in chapter for token in local_capture_tokens)

    assert "The re-profile then exposed normalization" in days[3]
    assert "Re-profiling then placed" in days[4]
    assert "Now repeat the exact baseline commands" in days[5]
    day6 = (ROOT / "book/src/week2-06-operator-lab.md").read_text()
    assert "pdm run profile-week2-kernels --solution tiny_llm" in day6
    assert "simd-matmul:decode:128" in day6
    assert "decode-attention:decode:128" in day6
    assert not any(token in day6 for token in local_capture_tokens)
    assert "Continue to [Day 7]" in day6


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


def test_matrix_defaults_cover_runnable_paired_context_points(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_course_progression.py",
            "--solution",
            "tiny_llm",
            "--suite",
            "week2",
            "--matrix",
            "--prefill-logits",
            "last",
        ],
    )

    args = progression.parse_args()

    assert args.prompt_length == [128, 512, 2048, 8192]
    assert args.output_len == 128
    assert args.variant == ["week2-simd-matmul", "mlx"]
    assert [
        progression.prompt_classification(value) for value in args.prompt_length
    ] == [
        "micro/regression",
        "product-context",
        "product-context",
        "product-context",
    ]


def test_matrix_accepts_explicit_mlx_only_native_endpoint(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_course_progression.py",
            "--solution",
            "tiny_llm",
            "--suite",
            "week2",
            "--matrix",
            "--prefill-logits",
            "last",
            "--variant",
            "mlx",
            "--prompt-length",
            "32640",
        ],
    )

    args = progression.parse_args()

    assert args.prompt_length == [32640]
    assert args.variant == ["mlx"]
    assert progression.prompt_classification(32640) == "native-product-endpoint"


def test_matrix_rejects_course_variant_at_native_endpoint_before_work(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_course_progression.py",
            "--solution",
            "tiny_llm",
            "--suite",
            "week2",
            "--matrix",
            "--prefill-logits",
            "last",
            "--prompt-length",
            "32640",
        ],
    )
    monkeypatch.setattr(
        progression,
        "collect_host_metadata",
        lambda: pytest.fail("host/model/timing work must not begin"),
    )

    with pytest.raises(SystemExit):
        progression.main()

    assert "32640 is available only for MLX" in capsys.readouterr().err


def test_matrix_accepts_day5_and_candidate_course_pair(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_course_progression.py",
            "--solution",
            "tiny_llm",
            "--suite",
            "week2",
            "--matrix",
            "--prefill-logits",
            "last",
            "--variant",
            "week2-simd-matmul",
            "--variant",
            "week2-io-aware-dense-attention",
        ],
    )

    args = progression.parse_args()

    assert args.variant == [
        "week2-simd-matmul",
        "week2-io-aware-dense-attention",
    ]


@pytest.mark.parametrize(
    ("extra_args", "message"),
    (
        (("--prompt-length", "512"), "requires --matrix"),
        (("--matrix", "--prefill-logits", "last"), "requires --suite week2"),
        (
            (
                "--suite",
                "week2",
                "--matrix",
                "--prefill-logits",
                "last",
                "--prompt-length",
                "100000",
            ),
            "invalid choice",
        ),
        (
            (
                "--suite",
                "week2",
                "--matrix",
                "--prefill-logits",
                "last",
                "--variant",
                "week2-simd-matmul",
            ),
            "exactly two Week 2 variants",
        ),
        (
            (
                "--suite",
                "week2",
                "--matrix",
                "--prefill-logits",
                "last",
                "--output-len",
                "129",
            ),
            "exactly --output-len 128",
        ),
    ),
)
def test_matrix_selector_negative_cases_fail_closed(
    monkeypatch, capsys, extra_args, message
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench_course_progression.py", "--solution", "tiny_llm", *extra_args],
    )

    with pytest.raises(SystemExit):
        progression.parse_args()

    assert message in capsys.readouterr().err


def test_product_metrics_and_spread_are_explicit_and_deterministic():
    samples = [
        progression.product_metrics(
            progression.Throughput(prefill=512.0, decode=50.0, output=40.0),
            512,
        ),
        progression.product_metrics(
            progression.Throughput(prefill=256.0, decode=40.0, output=30.0),
            512,
        ),
    ]

    summary = progression.summarize_product_metrics(samples)

    assert samples[0].TTFT_ms == 1000.0
    assert samples[0].TPOT_ms == 20.0
    assert summary["TTFT_ms"] == {
        "median": 1500.0,
        "minimum": 1000.0,
        "maximum": 2000.0,
        "spread": 1000.0,
    }
    assert summary["decode_tokens_per_second"]["spread"] == 10.0
    with pytest.raises(ValueError, match="positive"):
        progression.product_metrics(
            progression.Throughput(prefill=0.0, decode=1.0, output=1.0),
            128,
        )


def test_matrix_sample_uses_one_fresh_product_subprocess(monkeypatch, tmp_path):
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        raw_output = Path(command[command.index("--json-output") + 1])
        raw_output.write_text(
            json.dumps(
                {
                    "dispatch_counters": {"io_aware_dense_attention": 7},
                    "metrics": {
                        "output_tokens_per_second": 40.0,
                        "prefill_tokens_per_second": 512.0,
                        "decode_tokens_per_second": 50.0,
                    },
                }
            )
        )
        return SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        )

    args = SimpleNamespace(
        solution="tiny_llm",
        model="qwen3-4b",
        device="gpu",
        input_len=128,
        output_len=128,
        warmup=2,
        prefill_logits="last",
        seed=0,
        offline=True,
    )
    monkeypatch.setattr(progression.subprocess, "run", fake_run)

    result = progression.run_variant(
        tmp_path,
        progression.VARIANTS_BY_KEY["week2-simd-matmul"],
        args,
        input_len=8192,
    )

    assert result == progression.Throughput(
        prefill=512.0,
        decode=50.0,
        output=40.0,
        dispatch_counters={"io_aware_dense_attention": 7},
    )
    assert len(observed) == 1
    command, kwargs = observed[0]
    assert command[command.index("--min-input-len") + 1] == "8192"
    assert command[command.index("--max-input-len") + 1] == "8192"
    assert command[command.index("--min-output-len") + 1] == "128"
    assert command[command.index("--max-output-len") + 1] == "128"
    assert command[-2:] == ["--week2-checkpoint", "simd-matmul"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"


def test_matrix_json_schema_records_phase_metrics_and_fresh_process_order(
    tmp_path, monkeypatch
):
    output = tmp_path / "matrix.json"
    args = SimpleNamespace(
        json_output=output,
        model="qwen3-4b",
        solution="tiny_llm",
        suite="week2",
        device="gpu",
        input_len=128,
        output_len=128,
        warmup=1,
        repeats=2,
        seed=0,
        prefill_logits="last",
        offline=True,
        cooldown_seconds=0.0,
        variant=["week2-simd-matmul", "mlx"],
        prompt_length=[128, 8192],
        matrix=True,
    )
    calls = []

    def fake_run(_root, variant, _args, prompt_tokens):
        calls.append((prompt_tokens, variant.key))
        return progression.Throughput(
            prefill=float(prompt_tokens),
            decode=50.0,
            output=40.0,
            dispatch_counters={"io_aware_dense_attention": prompt_tokens},
        )

    monkeypatch.setattr(progression, "parse_args", lambda: args)
    monkeypatch.setattr(
        progression,
        "collect_host_metadata",
        lambda: {"platform": "test", "machine": "arm64", "mlx_version": "test"},
    )
    monkeypatch.setattr(
        progression,
        "collect_source_metadata",
        lambda _root: {"commit": "head", "tree": "tree", "tracked_dirty": False},
    )
    monkeypatch.setattr(progression, "run_variant", fake_run)

    progression.main()
    payload = json.loads(output.read_text())

    assert payload["schema_version"] == 3
    assert payload["evidence_kind"] == "single_request_product_matrix"
    assert payload["process_isolation"] == "fresh_process_per_sample"
    assert "no 100K+ product claim" in payload["context_boundary"]
    assert payload["workload"]["prompt_lengths"] == [128, 8192]
    assert payload["workload"]["decode_sample_tokens"] == 127
    assert payload["generated_tokens"] == 128
    assert payload["post_first_decode_intervals"] == 127
    assert payload["native_context_tokens"] == 32768
    assert payload["prompt_classification"] == {
        "128": "micro/regression",
        "8192": "product-context",
    }
    assert set(payload["metric_definitions"]) == {
        "TTFT_ms",
        "prefill_tokens_per_second",
        "TPOT_ms",
        "decode_tokens_per_second",
        "output_tokens_per_second",
    }
    assert len(payload["results"]) == 4
    assert all(len(result["samples"]) == 2 for result in payload["results"])
    assert all(
        "dispatch_counters" in sample
        for result in payload["results"]
        for sample in result["samples"]
    )
    assert len(calls) == 8
    assert calls[:4] == [
        (128, "week2-simd-matmul"),
        (128, "mlx"),
        (8192, "mlx"),
        (8192, "week2-simd-matmul"),
    ]


def test_single_point_json_keeps_schema_two_and_legacy_shape(tmp_path, monkeypatch):
    output = tmp_path / "single.json"
    args = SimpleNamespace(
        json_output=output,
        model="qwen3-0.6b",
        solution="tiny_llm",
        suite="week2",
        device="gpu",
        input_len=128,
        output_len=65,
        warmup=0,
        repeats=1,
        seed=0,
        prefill_logits="all",
        offline=True,
        cooldown_seconds=0.0,
        variant=["week2-simd-matmul"],
        prompt_length=[128],
        matrix=False,
    )
    monkeypatch.setattr(progression, "parse_args", lambda: args)
    monkeypatch.setattr(
        progression,
        "collect_host_metadata",
        lambda: {"platform": "test", "machine": "arm64", "mlx_version": "test"},
    )
    monkeypatch.setattr(
        progression,
        "collect_source_metadata",
        lambda _root: {"commit": "head", "tree": "tree", "tracked_dirty": False},
    )
    monkeypatch.setattr(
        progression,
        "run_variant",
        lambda *_args: progression.Throughput(10.0, 20.0, 15.0),
    )

    progression.main()
    payload = json.loads(output.read_text())

    assert payload["schema_version"] == 2
    assert "matrix" not in payload["configuration"]
    assert "prompt_lengths" not in payload["workload"]
    assert payload["execution_order"] == [["week2-simd-matmul"]]
    assert payload["results"]["week2-simd-matmul"]["median"] == {
        "prefill": 10.0,
        "decode": 20.0,
        "output": 15.0,
    }


def test_matrix_help_names_runnable_pair_and_mlx_only_endpoint(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bench_course_progression.py", "--help"])
    with pytest.raises(SystemExit) as exited:
        progression.parse_args()
    help_text = " ".join(capsys.readouterr().out.split())
    assert exited.value.code == 0
    assert "TTFT_ms" in help_text
    assert "TPOT_ms" in help_text
    assert "128, 512, 2048, and 8192" in help_text
    assert "32640 is available only with --variant mlx" in help_text
    assert "native 32768-token ceiling" in help_text


def test_public_main_help_lists_only_current_week2_checkpoints():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "io-aware-dense-attention" in result.stdout
    assert "shared-input-qkv" in result.stdout
    assert "shared-input-gate-up-swiglu" in result.stdout
    assert "decode-attention" not in result.stdout
    assert "split-k" not in result.stdout
    assert "long-context-attention" not in result.stdout
    assert "fused-gate-up" not in bench.WEEK2_CHECKPOINTS


@pytest.mark.parametrize(
    ("legacy", "replacement"),
    (
        ("decode-attention", "io-aware-dense-attention"),
        ("long-context-attention", "io-aware-dense-attention"),
        ("split-k", "shared-input-gate-up-swiglu"),
        ("fused-gate-up", "shared-input-gate-up-swiglu"),
    ),
)
def test_public_main_legacy_checkpoint_reports_migration(legacy, replacement):
    result = subprocess.run(
        [sys.executable, "main.py", "--week2-checkpoint", legacy],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"{legacy!r} was replaced by {replacement!r}" in result.stderr


def test_public_bench_help_lists_only_current_week2_checkpoints(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bench.py", "--help"])
    with pytest.raises(SystemExit) as exited:
        bench.parse_args()
    help_text = capsys.readouterr().out

    assert exited.value.code == 0
    assert "io-aware-dense-attention" in help_text
    assert "shared-input-qkv" in help_text
    assert "shared-input-gate-up-swiglu" in help_text
    assert "decode-attention" not in help_text
    assert "split-k" not in help_text
    assert "long-context-attention" not in help_text
    assert "fused-gate-up" not in bench.WEEK2_CHECKPOINTS
    assert "--disable-week2-io-aware-dense-attention" in help_text
    assert "--disable-week2-shared-input-qkv" in help_text
    assert "--disable-week2-shared-input-gate-up-swiglu" in help_text
    assert "--disable-week2-long-context-attention" not in help_text
    assert "--disable-week2-fused-gate-up" not in help_text


@pytest.mark.parametrize(
    ("legacy", "replacement"),
    (
        ("decode-attention", "io-aware-dense-attention"),
        ("long-context-attention", "io-aware-dense-attention"),
        ("split-k", "shared-input-gate-up-swiglu"),
        ("fused-gate-up", "shared-input-gate-up-swiglu"),
    ),
)
def test_public_bench_legacy_checkpoint_reports_migration(legacy, replacement):
    result = subprocess.run(
        [sys.executable, "-m", "benches.bench", "--week2-checkpoint", legacy],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"{legacy!r} was replaced by {replacement!r}" in result.stderr


@pytest.mark.parametrize(
    "checkpoint",
    (
        "shared-input-qkv",
        "shared-input-gate-up-swiglu",
        "io-aware-dense-attention",
    ),
)
def test_public_bench_accepts_current_checkpoint(monkeypatch, checkpoint):
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench.py", "--week2-checkpoint", checkpoint],
    )

    assert bench.parse_args().week2_checkpoint == checkpoint


def test_public_bench_unknown_checkpoint_lists_only_current_values():
    result = subprocess.run(
        [sys.executable, "-m", "benches.bench", "--week2-checkpoint", "unknown"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unknown Week 2 checkpoint 'unknown'" in result.stderr
    assert "io-aware-dense-attention" in result.stderr
    assert "shared-input-qkv" in result.stderr
    assert "shared-input-gate-up-swiglu" in result.stderr
    assert "decode-attention" not in result.stderr
    assert "split-k" not in result.stderr
    assert "long-context-attention" not in result.stderr
    assert "fused-gate-up" not in bench.WEEK2_CHECKPOINTS
