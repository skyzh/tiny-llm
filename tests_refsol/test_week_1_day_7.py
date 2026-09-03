import mlx.core as mx
import numpy as np
import pytest

from .tiny_llm_base import make_sampler


def log_probabilities(probabilities: list[float], dtype=mx.float32) -> mx.array:
    return mx.log(mx.array([probabilities], dtype=dtype))


def assert_token(token: mx.array, expected: int):
    assert token.shape == (1,)
    assert token.dtype == mx.uint32
    np.testing.assert_array_equal(np.array(token), [expected])


def categorical_result(expected: np.ndarray, token: int):
    def categorical(logits: mx.array, axis: int) -> mx.array:
        assert axis == -1
        actual = np.array(logits)
        actual_probabilities = np.exp(actual - np.max(actual, axis=-1, keepdims=True))
        actual_probabilities /= actual_probabilities.sum(axis=-1, keepdims=True)
        expected_probabilities = np.exp(
            expected - np.max(expected, axis=-1, keepdims=True)
        )
        expected_probabilities /= expected_probabilities.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(
            actual_probabilities, expected_probabilities, atol=1e-6
        )
        return mx.array([token], dtype=mx.uint32)

    return categorical


def test_task_1_greedy_shape_dtype_preserves_input_without_rng(monkeypatch):
    logprobs = log_probabilities([0.1, 0.6, 0.3], dtype=mx.bfloat16)
    before = np.array(logprobs.astype(mx.float32))

    def categorical_forbidden(*args, **kwargs):
        raise AssertionError("greedy sampling consumed random state")

    monkeypatch.setattr(mx.random, "categorical", categorical_forbidden)
    token = make_sampler(temp=0, top_p=0.1, top_k=1)(logprobs)

    assert_token(token, 1)
    np.testing.assert_array_equal(np.array(logprobs.astype(mx.float32)), before)


def test_task_1_temperature_divides_log_probabilities(monkeypatch):
    logprobs = log_probabilities([0.1, 0.2, 0.7])
    before = np.array(logprobs)
    expected = before / 0.5
    monkeypatch.setattr(mx.random, "categorical", categorical_result(expected, 2))

    token = make_sampler(temp=0.5, top_p=None, top_k=None)(logprobs)

    assert_token(token, 2)
    np.testing.assert_array_equal(np.array(logprobs), before)


def test_task_1_top_k_keeps_exactly_the_largest_k(monkeypatch):
    logprobs = log_probabilities([0.1, 0.4, 0.2, 0.3])
    before = np.array(logprobs)
    expected = before.copy()
    expected[:, [0, 2]] = -np.inf
    monkeypatch.setattr(mx.random, "categorical", categorical_result(expected, 1))

    token = make_sampler(temp=1.0, top_p=None, top_k=2)(logprobs)

    assert_token(token, 1)
    np.testing.assert_array_equal(np.array(logprobs), before)


def test_task_1_top_p_keeps_the_threshold_crossing_token(monkeypatch):
    logprobs = log_probabilities([0.5, 0.3, 0.15, 0.05])
    before = np.array(logprobs)
    expected = before.copy()
    expected[:, 2:] = -np.inf
    monkeypatch.setattr(mx.random, "categorical", categorical_result(expected, 0))

    token = make_sampler(temp=1.0, top_p=0.6, top_k=None)(logprobs)

    assert_token(token, 0)
    np.testing.assert_array_equal(np.array(logprobs), before)


def test_task_1_composes_top_k_then_top_p_then_temperature(monkeypatch):
    logprobs = log_probabilities([0.45, 0.30, 0.15, 0.10])
    before = np.array(logprobs)
    expected = before.copy()
    expected[:, 2:] = -np.inf
    expected /= 0.5
    monkeypatch.setattr(mx.random, "categorical", categorical_result(expected, 1))

    token = make_sampler(temp=0.5, top_p=0.7, top_k=3)(logprobs)

    assert_token(token, 1)
    np.testing.assert_array_equal(np.array(logprobs), before)


def test_task_1_none_disables_both_filters(monkeypatch):
    logprobs = log_probabilities([0.2, 0.3, 0.5])
    expected = np.array(logprobs)
    monkeypatch.setattr(mx.random, "categorical", categorical_result(expected, 2))

    assert_token(make_sampler(1.0, None, None)(logprobs), 2)


def test_task_1_non_positive_values_disable_both_filters(monkeypatch):
    logprobs = log_probabilities([0.2, 0.3, 0.5])
    expected = np.array(logprobs)
    monkeypatch.setattr(mx.random, "categorical", categorical_result(expected, 2))

    assert_token(make_sampler(1.0, 0.0, 0)(logprobs), 2)


def test_task_1_full_vocabulary_boundaries_are_unfiltered(monkeypatch):
    logprobs = log_probabilities([0.1, 0.2, 0.3, 0.4])
    expected = np.array(logprobs)
    monkeypatch.setattr(mx.random, "categorical", categorical_result(expected, 3))

    assert_token(make_sampler(1.0, 1.0, 4)(logprobs), 3)


def test_task_1_oversized_top_k_preserves_existing_error():
    logprobs = log_probabilities([0.1, 0.2, 0.3, 0.4])

    with pytest.raises(ValueError):
        mx.eval(make_sampler(1.0, None, 5)(logprobs))


def test_task_1_seeded_categorical_sampling_is_repeatable():
    logprobs = log_probabilities([0.05, 0.15, 0.3, 0.5])
    sampler = make_sampler(0.8, 0.95, 4)

    mx.random.seed(2026)
    first = sampler(logprobs)
    mx.eval(first)
    mx.random.seed(2026)
    second = sampler(logprobs)
    mx.eval(second)

    assert first.shape == (1,)
    assert first.dtype == mx.uint32
    np.testing.assert_array_equal(np.array(first), np.array(second))
