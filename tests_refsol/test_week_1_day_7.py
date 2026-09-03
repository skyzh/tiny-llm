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


def sampled_tokens(
    sampler, logprobs: mx.array, *, draws: int = 384, seed: int = 2026
) -> np.ndarray:
    tokens = np.empty(draws, dtype=np.int64)
    mx.random.seed(seed)
    for index in range(draws):
        token = sampler(logprobs)
        mx.eval(token)
        assert token.shape == (1,)
        assert token.dtype == mx.uint32
        tokens[index] = int(np.array(token).item())
    return tokens


def sampled_frequencies(
    sampler, logprobs: mx.array, *, draws: int = 384, seed: int = 2026
) -> np.ndarray:
    tokens = sampled_tokens(sampler, logprobs, draws=draws, seed=seed)
    counts = np.bincount(tokens, minlength=logprobs.shape[-1])
    return counts / draws


def assert_distribution(
    actual: np.ndarray, expected: list[float], *, atol: float = 0.08
):
    expected_array = np.array(expected, dtype=np.float64)
    expected_array /= expected_array.sum()
    np.testing.assert_array_equal(actual[expected_array == 0], 0.0)
    np.testing.assert_allclose(actual, expected_array, atol=atol, rtol=0)


def test_task_1_greedy_shape_dtype_preserves_input_and_rng_state():
    logprobs = log_probabilities([0.1, 0.6, 0.3], dtype=mx.bfloat16)
    before = np.array(logprobs.astype(mx.float32))

    mx.random.seed(2026)
    expected_next_draw = mx.random.uniform(shape=(8,))
    mx.eval(expected_next_draw)

    mx.random.seed(2026)
    token = make_sampler(temp=0, top_p=0.1, top_k=1)(logprobs)
    mx.eval(token)
    actual_next_draw = mx.random.uniform(shape=(8,))
    mx.eval(actual_next_draw)

    assert_token(token, 1)
    np.testing.assert_array_equal(np.array(logprobs.astype(mx.float32)), before)
    np.testing.assert_array_equal(
        np.array(actual_next_draw), np.array(expected_next_draw)
    )


def test_task_1_temperature_divides_log_probabilities():
    probabilities = [0.7, 0.2, 0.1]
    logprobs = log_probabilities(probabilities)
    before = np.array(logprobs)

    frequencies = sampled_frequencies(
        make_sampler(temp=0.5, top_p=None, top_k=None), logprobs
    )

    assert_distribution(frequencies, [probability**2 for probability in probabilities])
    np.testing.assert_array_equal(np.array(logprobs), before)


def test_task_1_top_k_keeps_exactly_the_largest_k():
    logprobs = log_probabilities([0.1, 0.4, 0.2, 0.3])
    before = np.array(logprobs)

    frequencies = sampled_frequencies(
        make_sampler(temp=1.0, top_p=None, top_k=2), logprobs
    )

    assert_distribution(frequencies, [0.0, 0.4, 0.0, 0.3])
    np.testing.assert_array_equal(np.array(logprobs), before)


def test_task_1_top_p_keeps_the_threshold_crossing_token():
    logprobs = log_probabilities([0.5, 0.3, 0.15, 0.05])
    before = np.array(logprobs)

    frequencies = sampled_frequencies(
        make_sampler(temp=1.0, top_p=0.6, top_k=None), logprobs
    )

    assert_distribution(frequencies, [0.5, 0.3, 0.0, 0.0])
    np.testing.assert_array_equal(np.array(logprobs), before)


def test_task_1_composes_top_k_then_top_p_then_temperature():
    logprobs = log_probabilities([0.55, 0.25, 0.15, 0.05])
    before = np.array(logprobs)

    frequencies = sampled_frequencies(
        make_sampler(temp=0.5, top_p=0.7, top_k=3), logprobs
    )

    assert_distribution(frequencies, [0.55**2, 0.25**2, 0.0, 0.0])
    np.testing.assert_array_equal(np.array(logprobs), before)


def test_task_1_none_disables_both_filters():
    logprobs = log_probabilities([0.1, 0.2, 0.3, 0.4])

    frequencies = sampled_frequencies(make_sampler(1.0, None, None), logprobs)

    assert_distribution(frequencies, [0.1, 0.2, 0.3, 0.4])


def test_task_1_non_positive_values_disable_both_filters():
    logprobs = log_probabilities([0.1, 0.2, 0.3, 0.4])

    frequencies = sampled_frequencies(make_sampler(1.0, 0.0, 0), logprobs)

    assert_distribution(frequencies, [0.1, 0.2, 0.3, 0.4])


def test_task_1_full_vocabulary_boundaries_are_unfiltered():
    logprobs = log_probabilities([0.1, 0.2, 0.3, 0.4])

    frequencies = sampled_frequencies(make_sampler(1.0, 1.0, 4), logprobs)

    assert_distribution(frequencies, [0.1, 0.2, 0.3, 0.4])


def test_task_1_oversized_top_k_preserves_existing_error():
    logprobs = log_probabilities([0.1, 0.2, 0.3, 0.4])

    with pytest.raises(ValueError):
        mx.eval(make_sampler(1.0, None, 5)(logprobs))


def test_task_1_seeded_sampling_repeats_and_changes_with_seed():
    logprobs = log_probabilities([0.05, 0.15, 0.3, 0.5])

    first = sampled_tokens(make_sampler(0.8, 0.95, 4), logprobs, draws=32)
    second = sampled_tokens(make_sampler(0.8, 0.95, 4), logprobs, draws=32)
    different_seed = sampled_tokens(
        make_sampler(0.8, 0.95, 4), logprobs, draws=32, seed=2027
    )

    np.testing.assert_array_equal(first, second)
    assert np.any(first != different_seed)
