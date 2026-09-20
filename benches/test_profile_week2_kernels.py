import argparse
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest

from benches import profile_week2_kernels as profile


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
        assert implementation.decode_attention_max_query == 2
        assert implementation.decode_attention_max_context == 32768


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
    assert result["evidence_kind"] == "synchronized_operator_replay"
    assert "not production traffic share" in result["interpretation"]

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
    checkpoints = [case.checkpoint for case in cases]
    assert checkpoints.index("simd-matmul") < checkpoints.index("shared-input-qkv")
    assert checkpoints[-1] == "io-aware-dense-attention"

    for implementation_name in ("tiny_llm", "tiny_llm_ref"):
        implementation = profile.load_implementation(implementation_name)
        assert implementation.checkpoints == (
            "kv-cache",
            "quantized-matvec",
            "rmsnorm",
            "rope",
            "swiglu",
            "simd-matmul",
            "shared-input-qkv",
            "shared-input-gate-up-swiglu",
            "io-aware-dense-attention",
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


def test_component_taxonomy_is_model_level_and_stable():
    assert profile.COMPONENT_CATEGORIES == (
        "attention_q_projection",
        "attention_k_projection",
        "attention_v_projection",
        "attention_o_projection",
        "attention_core",
        "mlp_gate_projection",
        "mlp_up_projection",
        "mlp_down_projection",
        "swiglu",
        "norms_rope_residuals",
        "vocabulary_head_embedding",
        "kv_cache_growth",
    )
    assert profile.EVIDENCE_KIND == "synchronized_operator_replay"
    assert "not production traffic share" in profile.EVIDENCE_BOUNDARY


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("simd-matmul:training:128", "phase must be decode or prefill"),
        ("simd-matmul:prefill:0", "tokens must be positive"),
        ("simd-matmul:prefill", "CHECKPOINT:PHASE:TOKENS"),
    ),
)
def test_component_case_selector_nearest_negatives(value, message):
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        profile.parse_case(value)


def test_profile_case_emits_every_component_with_phase_and_context(monkeypatch):
    class FakeReplay:
        def __init__(self, *_args):
            pass

    for category in profile.COMPONENT_CATEGORIES:
        setattr(FakeReplay, category, lambda self: [])

    observed = []

    def fake_benchmark(builders, warmup, iterations):
        observed.extend(name for name, _build in builders)
        assert warmup == 2
        assert iterations == 4
        return {name: float(index + 1) for index, (name, _build) in enumerate(builders)}

    implementation = SimpleNamespace(model_type=lambda *_args, **_kwargs: object())
    monkeypatch.setattr(profile, "KernelReplay", FakeReplay)
    monkeypatch.setattr(profile, "benchmark_groups", fake_benchmark)

    result = profile.profile_case(
        implementation,
        object(),
        "Qwen/Qwen3-4B-MLX-4bit",
        profile.ProfileCase("simd-matmul", "prefill", 2048),
        warmup=2,
        iterations=4,
    )

    assert observed == list(profile.COMPONENT_CATEGORIES)
    assert result["phase"] == "prefill"
    assert result["context_tokens"] == 2048
    assert result["evidence_kind"] == "synchronized_operator_replay"
    assert "not production traffic share" in result["interpretation"]
    assert [category["name"] for category in result["categories"]] == list(
        profile.COMPONENT_CATEGORIES
    )
    assert sum(category["share"] for category in result["categories"]) == pytest.approx(
        1.0
    )


def test_component_replay_keeps_legacy_method_names():
    for method in ("projections", "attention", "pointwise", "cache"):
        assert callable(getattr(profile.KernelReplay, method))


def test_component_replay_builders_execute_with_model_level_shapes():
    class Embedding:
        def __call__(self, tokens):
            return mx.zeros((*tokens.shape, 4), dtype=mx.float32)

        def as_linear(self, hidden):
            return hidden @ mx.ones((4, 5), dtype=mx.float32)

    attention = SimpleNamespace(
        num_kv_heads=1,
        head_dim=2,
        scale=0.5,
        use_decode_attention=False,
        use_fast_rope=False,
        wq=mx.eye(4),
        wk=mx.ones((4, 2)),
        wv=mx.ones((4, 2)),
        wo=mx.eye(4),
        q_norm=lambda value: value,
        k_norm=lambda value: value,
        rope=lambda value, *, offset: value,
    )
    mlp = SimpleNamespace(
        hidden_dim=6,
        use_fast_swiglu=False,
        w_gate=mx.ones((4, 6)),
        w_up=mx.ones((4, 6)),
        w_down=mx.ones((6, 4)),
    )
    layer = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        self_attn=attention,
        mlp=mlp,
        input_layernorm=lambda value: value,
        post_attention_layernorm=lambda value: value,
    )
    implementation = SimpleNamespace(
        quantized_weights_type=type(None),
        linear=lambda value, weight: value @ weight,
        quantized_linear=lambda *_args: pytest.fail("unexpected quantized path"),
        grouped_attention=lambda query, *_args, **_kwargs: query,
        decode_attention=lambda query, *_args, **_kwargs: query,
        decode_attention_max_query=0,
        decode_attention_max_context=0,
        silu=lambda value: value,
        swiglu=lambda gate, up: gate * up,
    )
    model = SimpleNamespace(
        layers_inner=[layer],
        precision=mx.float32,
        embedding=Embedding(),
        w_lm_head=None,
        norm=lambda value: value,
    )

    for phase in ("decode", "prefill"):
        replay = profile.KernelReplay(implementation, model, phase, 8)
        for category in profile.COMPONENT_CATEGORIES:
            outputs = getattr(replay, category)()
            assert outputs
            mx.eval(*outputs)


def test_profile_help_states_operator_evidence_boundary(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["profile_week2_kernels.py", "--help"])
    with pytest.raises(SystemExit) as exited:
        profile.parse_args()
    help_text = capsys.readouterr().out
    assert exited.value.code == 0
    assert "synchronized operator evidence" in help_text
    assert "not production traffic share" in help_text
    for category in profile.COMPONENT_CATEGORIES:
        assert category in help_text
