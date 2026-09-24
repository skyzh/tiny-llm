from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from benches import profile_week2_kernels as profile
from tests_refsol.utils import tiny_qwen3_mlx_model


def test_kernel_group_profile_rotates_every_group_through_each_position():
    calls = []

    def build(name):
        def run():
            calls.append(name)
            return []

        return run

    builders = tuple((name, build(name)) for name in ("a", "b", "c", "d"))

    profile.benchmark_groups(builders, warmup=0, iterations=4)

    assert calls == [
        "a",
        "b",
        "c",
        "d",
        "b",
        "c",
        "d",
        "a",
        "c",
        "d",
        "a",
        "b",
        "d",
        "a",
        "b",
        "c",
    ]


def test_profile_reuses_the_model_decode_attention_boundaries():
    implementation = SimpleNamespace(
        decode_attention_max_query=2,
        decode_attention_max_context=256,
    )
    explicit_mask = mx.zeros((1, 1, 1, 1), dtype=mx.float32)

    cases = (
        (True, 1, 1, None, True),
        (True, 2, 256, "causal", True),
        (True, 3, 256, None, False),
        (True, 2, 257, None, False),
        (True, 1, 1, explicit_mask, False),
        (False, 1, 1, None, False),
    )
    for enabled, query_length, context_length, mask, expected in cases:
        assert (
            profile.should_use_decode_attention(
                implementation,
                enabled,
                query_length,
                context_length,
                mask,
            )
            is expected
        )


@pytest.mark.parametrize(
    ("phase", "tokens", "enabled", "expected_path", "expected_mask"),
    (
        ("decode", 128, True, "custom", None),
        ("decode", 256, True, "custom", None),
        ("decode", 257, True, "readable", None),
        ("prefill", 2, True, "custom", "causal"),
        ("prefill", 3, True, "readable", "causal"),
        ("decode", 128, False, "readable", None),
    ),
)
def test_kernel_replay_routes_attention_with_production_guard(
    phase,
    tokens,
    enabled,
    expected_path,
    expected_mask,
):
    calls = []

    def record(path):
        def attention(query, _key, _value, *, scale, mask):
            calls.append((path, scale, mask))
            return query

        return attention

    implementation = SimpleNamespace(
        decode_attention=record("custom"),
        grouped_attention=record("readable"),
        decode_attention_max_query=2,
        decode_attention_max_context=256,
    )
    attention = SimpleNamespace(
        num_kv_heads=1,
        head_dim=4,
        scale=0.5,
        use_decode_attention=enabled,
    )
    layer = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=1,
        self_attn=attention,
        mlp=SimpleNamespace(hidden_dim=8),
    )
    model = SimpleNamespace(layers_inner=[layer], precision=mx.float32)

    replay = profile.KernelReplay(implementation, model, phase, tokens)
    replay.attention()

    assert calls == [(expected_path, 0.5, expected_mask)]


def test_student_and_reference_profiles_share_the_production_guard():
    for name in ("tiny_llm", "tiny_llm_ref"):
        implementation = profile.load_implementation(name)
        assert implementation.decode_attention_max_query == 0
        assert implementation.decode_attention_max_context == 0


def test_selected_prefill_attribution_uses_the_tiled_operator():
    implementation = profile.load_implementation("tiny_llm_ref")
    model = implementation.model_type(
        tiny_qwen3_mlx_model(head_dim=128), checkpoint="selected"
    )
    calls = []

    def record(query, _key, _value, *, scale, mask):
        calls.append((query.shape[-2], scale, mask))
        return query

    replay = profile.KernelReplay(
        replace(implementation, tiled_attention=record), model, "prefill", 9
    )
    outputs = replay.attention()
    assert len(outputs) == len(model.layers_inner)
    assert calls == [(9, model.layers_inner[0].self_attn.scale, "causal")]


def test_decision_requires_exact_source_solution_model_and_workload_identity():
    baseline = {
        "source": {"tree": "tree"},
        "solution": "tiny_llm",
        "model": "model",
        "checkpoint": "swiglu",
        "workload_id": "workload",
    }
    candidate = {
        **baseline,
        "checkpoint": "simd-matmul",
    }
    result = profile.build_decision(
        baseline,
        candidate,
        dominant_category="projections",
        hypothesis="SIMD prefill reduces projection time",
        observed_effect="candidate reduced matched product and attribution time",
        decision="keep",
        next_experiment="profile the next dominant category",
    )
    assert result["baseline_checkpoint"] == "swiglu"
    assert result["candidate_checkpoint"] == "simd-matmul"
    assert result["decision"] == "keep"

    for field in ("source", "solution", "model", "workload_id"):
        mismatch = dict(candidate)
        mismatch[field] = "different"
        with pytest.raises(ValueError, match=field.replace("_", ".*")):
            profile.build_decision(
                baseline,
                mismatch,
                dominant_category="projections",
                hypothesis="hypothesis",
                observed_effect="effect",
                decision="inconclusive",
                next_experiment="next",
            )


def test_profile_workload_identity_covers_every_workload_field(monkeypatch):
    baseline = profile.ProfileCase("swiglu", "prefill", 128)
    candidate = profile.ProfileCase("simd-matmul", "prefill", 128)
    original = profile.workload_record("model", baseline, 2, 4)
    assert original == profile.workload_record("model", candidate, 2, 4)

    variants = [
        profile.workload_record("other-model", baseline, 2, 4),
        profile.workload_record(
            "model", profile.ProfileCase("swiglu", "decode", 128), 2, 4
        ),
        profile.workload_record(
            "model", profile.ProfileCase("swiglu", "prefill", 64), 2, 4
        ),
        profile.workload_record("model", baseline, 3, 4),
        profile.workload_record("model", baseline, 2, 5),
    ]
    with monkeypatch.context() as patch:
        patch.setattr(profile, "PROMPT_RULE", "other-prompt-rule")
        variants.append(profile.workload_record("model", baseline, 2, 4))
    with monkeypatch.context() as patch:
        patch.setattr(profile, "PREFILL_LOGITS", "last")
        variants.append(profile.workload_record("model", baseline, 2, 4))

    original_hash = profile.canonical_hash(original)
    assert all(profile.canonical_hash(variant) != original_hash for variant in variants)
    assert [
        {key for key in original if original[key] != variant[key]}
        for variant in variants
    ] == [
        {"model"},
        {"phase"},
        {"tokens"},
        {"warmup"},
        {"iterations"},
        {"prompt_rule"},
        {"prefill_logits"},
    ]


def test_default_attribution_and_model_expose_only_canonical_checkpoints():
    cases = [profile.parse_case(value) for value in profile.DEFAULT_CASES]
    checkpoints = tuple(case.checkpoint for case in cases)
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
    assert {"decode-attention", "split-k"}.isdisjoint(checkpoints)

    for implementation_name in ("tiny_llm", "tiny_llm_ref"):
        implementation = profile.load_implementation(implementation_name)
        assert implementation.checkpoints == (
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


def test_profile_refuses_existing_output_before_loading_implementation(
    tmp_path, monkeypatch
):
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    monkeypatch.setattr(
        profile,
        "parse_args",
        lambda: SimpleNamespace(json_output=output, decision_output=None),
    )
    monkeypatch.setattr(
        profile,
        "load_implementation",
        lambda _name: pytest.fail("implementation/model work must not begin"),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        profile.main()


@pytest.mark.parametrize("alias_kind", ("exact", "lexical", "symlink-parent"))
def test_profile_refuses_output_alias_before_loading_implementation(
    tmp_path, monkeypatch, alias_kind
):
    output = tmp_path / "real" / "result.json"
    if alias_kind == "exact":
        alias = output
    elif alias_kind == "lexical":
        alias = output.parent / "unused" / ".." / output.name
    else:
        output.parent.mkdir()
        linked_parent = tmp_path / "linked"
        linked_parent.symlink_to(output.parent, target_is_directory=True)
        alias = linked_parent / output.name
    monkeypatch.setattr(
        profile,
        "parse_args",
        lambda: SimpleNamespace(json_output=output, decision_output=alias),
    )
    monkeypatch.setattr(
        profile,
        "load_implementation",
        lambda _name: pytest.fail("implementation/model work must not begin"),
    )
    with pytest.raises(ValueError, match="distinct"):
        profile.main()


def test_profile_preserves_distinct_output_role_order(tmp_path):
    json_output = tmp_path / "profile.json"
    decision_output = tmp_path / "decision.json"
    assert profile.normalize_output_paths(json_output, decision_output) == (
        json_output.resolve(),
        decision_output.resolve(),
    )
