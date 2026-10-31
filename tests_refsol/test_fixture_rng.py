"""Deterministic tiny-model fixtures must not perturb another test's RNG."""

import mlx.core as mx
import pytest

from .utils import tiny_qwen3_mlx_model


def _packed_parameters(fixture):
    yield fixture.model.embed_tokens
    for layer in fixture.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            yield getattr(layer.self_attn, name)
        for name in ("gate_proj", "up_proj", "down_proj"):
            yield getattr(layer.mlp, name)


@pytest.mark.parametrize("seed", (0, 4))
@pytest.mark.parametrize("head_dim", (64, 128))
def test_seeded_fixture_is_repeatable_and_preserves_global_rng(seed, head_dim):
    before = mx.random.state[0]
    first = tiny_qwen3_mlx_model(head_dim=head_dim, seed=seed)
    second = tiny_qwen3_mlx_model(head_dim=head_dim, seed=seed)

    assert bool(mx.array_equal(mx.random.state[0], before).item())
    for left, right in zip(_packed_parameters(first), _packed_parameters(second)):
        for name in ("weight", "scales", "biases"):
            assert bool(
                mx.array_equal(getattr(left, name), getattr(right, name)).item()
            )


def test_seeded_fixture_preserves_global_rng_on_failure(monkeypatch):
    before = mx.random.state[0]

    def fail_quantization(*args, **kwargs):
        raise RuntimeError("injected fixture failure")

    monkeypatch.setattr(mx, "quantize", fail_quantization)
    with pytest.raises(RuntimeError, match="injected fixture failure"):
        tiny_qwen3_mlx_model(seed=0)

    assert bool(mx.array_equal(mx.random.state[0], before).item())


def test_unseeded_fixture_keeps_global_rng_behavior():
    before = mx.random.state[0]
    tiny_qwen3_mlx_model()
    assert not bool(mx.array_equal(mx.random.state[0], before).item())


def test_seed_changes_fixture_without_advancing_global_rng():
    before = mx.random.state[0]
    first = tiny_qwen3_mlx_model(seed=0)
    second = tiny_qwen3_mlx_model(seed=4)
    assert not bool(
        mx.array_equal(
            first.model.embed_tokens.weight, second.model.embed_tokens.weight
        ).item()
    )
    assert bool(mx.array_equal(mx.random.state[0], before).item())
