from types import SimpleNamespace

import mlx.core as mx

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


def test_student_and_reference_profiles_share_the_production_guard():
    for name in ("tiny_llm", "tiny_llm_ref"):
        implementation = profile.load_implementation(name)
        assert implementation.decode_attention_max_query == 2
        assert implementation.decode_attention_max_context == 256
