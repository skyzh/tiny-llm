"""Week 3 Day 2 chunked-prefill tests."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

if Path(__file__).parent.name == "tests_refsol":
    import tiny_llm_ref.batch as batch_runtime
    from tiny_llm_ref.attention import causal_mask
else:
    import tiny_llm.batch as batch_runtime
    from tiny_llm.attention import causal_mask


class FakeDetokenizer:
    def __init__(self, _):
        self.text = ""

    def add_token(self, token):
        self.text += str(token)


class FakeTokenizer:
    eos_token_id = 99
    _tokenizer = object()
    detokenizer = FakeDetokenizer(_tokenizer)

    def encode(self, prompt, add_special_tokens=False):
        assert not add_special_tokens
        named_prompts = {
            "active": [10],
            "long": [20, 21, 22, 23, 24],
            "eos": [40],
            "at-limit": [50, 51],
            "over-limit": [60, 61, 62],
        }
        return named_prompts.get(prompt, list(range(1, len(prompt) + 1)))


class CacheReleaseProbe:
    def __init__(self):
        self.live_count = 0

    def assert_all_released(self):
        assert self.live_count == 0


class DenseCacheDouble:
    """Storage-neutral public cache lifecycle double."""

    def __init__(self, release_probe=None, fail_materialize=False):
        self.key_values = None
        self.offset = 0
        self.materialized = True
        self.released = False
        self.release_probe = release_probe
        self.fail_materialize = fail_materialize
        if self.release_probe is not None:
            self.release_probe.live_count += 1

    def update_and_fetch(self, key, value, mask_length=None, mask=None):
        assert not self.released
        assert key.shape == value.shape
        if self.key_values is None:
            keys, values = key, value
        else:
            keys = mx.concat([self.key_values[0], key], axis=2)
            values = mx.concat([self.key_values[1], value], axis=2)
        self.key_values = (keys, values)
        self.offset = keys.shape[2]
        self.materialized = False
        return keys, values, self.offset, mask

    def materialize(self):
        self.materialized = True
        if self.fail_materialize:
            raise RuntimeError("injected materialization failure")

    def release(self):
        if self.released:
            return
        self.released = True
        if self.release_probe is not None:
            self.release_probe.live_count -= 1


class ChunkBoundaryModel:
    num_hidden_layers = 2

    def create_kv_cache(self):
        return [DenseCacheDouble() for _ in range(self.num_hidden_layers)]

    def __call__(self, inputs, offsets, cache, logits_to_keep=1):
        assert logits_to_keep == 1
        offset = offsets[0] if isinstance(offsets, list) else int(offsets)
        assert all(layer.offset == offset for layer in cache)
        assert all(layer.materialized for layer in cache)

        values = inputs.astype(mx.float32).reshape(1, 1, inputs.shape[1], 1)
        for layer in cache:
            layer.update_and_fetch(values, values)

        next_token = int(inputs[0, -1].item()) + 1
        logits = mx.zeros((1, 1, 128), dtype=mx.float32)
        return logits.at[..., next_token].add(1)


@pytest.mark.parametrize(
    ("prompt_length", "expected_offsets"),
    [(2, [2]), (3, [3]), (4, [3, 4])],
    ids=["short", "exact", "step-plus-one"],
)
def test_chunk_boundaries_use_absolute_offsets_and_materialize_every_layer(
    prompt_length, expected_offsets
):
    request = batch_runtime.Request(
        ChunkBoundaryModel(),
        FakeTokenizer(),
        "x" * prompt_length,
        prefill_max_step=3,
    )

    for expected_offset in expected_offsets:
        request.try_prefill()
        assert request.offset == expected_offset
        assert all(layer.materialized for layer in request.kv_cache)

    assert request.is_prefill_done
    assert request.next_token == prompt_length + 1
    with pytest.raises(ValueError):
        request.try_prefill()


def test_nonzero_prefix_rectangular_mask_matches_one_shot_attention():
    values = mx.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=mx.float32)
    mask = causal_mask(L=2, S=5, dtype=mx.float32)
    visible = mx.exp(mask)
    chunked = mx.matmul(visible, values[:, None])[:, 0] / mx.sum(visible, axis=-1)
    one_shot_final = mx.mean(values)

    expected_mask = mx.array(
        [
            [0.0, 0.0, 0.0, 0.0, -mx.inf],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=mx.float32,
    )
    np.testing.assert_allclose(np.array(mask), np.array(expected_mask))
    np.testing.assert_allclose(np.array(chunked), np.array([5.0, 6.0]))
    np.testing.assert_allclose(np.array(chunked[-1]), np.array(one_shot_final))


class SchedulerModel:
    num_hidden_layers = 2

    def __init__(self, release_probe=None, fail_at=None):
        self.fail_at = fail_at
        self.release_probe = release_probe
        self.decode_advances = 0

    def create_kv_cache(self):
        group = [
            DenseCacheDouble(
                release_probe=self.release_probe,
                fail_materialize=self.fail_at == "materialize" and layer == 0,
            )
            for layer in range(self.num_hidden_layers)
        ]
        return group

    def __call__(self, inputs, offsets, cache, logits_to_keep=1):
        assert logits_to_keep == 1
        # Request prefill is a one-row call. This fake uses a two-slot batch so
        # decode is distinguishable without depending on a concrete cache type.
        is_decode = inputs.shape[0] == 2
        if self.fail_at == "prefill" and not is_decode:
            raise RuntimeError("injected prefill failure")
        if self.fail_at == "decode" and is_decode:
            raise RuntimeError("injected decode failure")

        sequence_length = 1 if is_decode else inputs.shape[1]
        values = inputs.astype(mx.float32).reshape(
            inputs.shape[0], 1, sequence_length, 1
        )
        for layer in cache:
            layer.update_and_fetch(values, values, mask_length=sequence_length)

        input_tokens = inputs[:, -1].tolist()
        if is_decode:
            self.decode_advances += sum(token != 0 for token in input_tokens)

        next_by_token = {
            0: 99,
            1: 2,
            2: 3,
            3: 4,
            4: 99,
            5: 99,
            10: 1,
            20: 1,
            21: 1,
            22: 1,
            23: 1,
            40: 99,
            51: 7,
        }
        logits = mx.zeros((inputs.shape[0], 1, 100), dtype=mx.float32)
        for row, token in enumerate(input_tokens):
            if token == 24:
                next_token = 99 if self.decode_advances >= 3 else 88
            else:
                next_token = next_by_token.get(token, 99)
            logits[row, 0, next_token] = 1
        return logits


def test_active_decode_advances_between_long_prompt_chunks_and_results_are_ordered(
    capsys,
):
    release_probe = CacheReleaseProbe()
    model = SchedulerModel(release_probe=release_probe)

    result = batch_runtime.batch_generate(
        model,
        FakeTokenizer(),
        ["active", "long"],
        max_seq_len=8,
        batch_size=2,
        prefill_step=2,
    )

    assert result == [(1, ""), (0, "1234")]
    release_probe.assert_all_released()
    capsys.readouterr()


@pytest.mark.parametrize(
    ("prompt", "max_seq_len"),
    [("eos", 4), ("at-limit", 2)],
    ids=["immediate-eos", "exact-max-length"],
)
def test_normal_completion_releases_every_request_cache(capsys, prompt, max_seq_len):
    release_probe = CacheReleaseProbe()
    model = SchedulerModel(release_probe=release_probe)

    result = batch_runtime.batch_generate(
        model,
        FakeTokenizer(),
        [prompt],
        max_seq_len=max_seq_len,
        batch_size=2,
        prefill_step=2,
    )

    assert result == [(0, "")]
    release_probe.assert_all_released()
    capsys.readouterr()


def test_oversized_prompt_leaves_no_owned_cache_live(capsys):
    release_probe = CacheReleaseProbe()
    model = SchedulerModel(release_probe=release_probe)

    with pytest.raises(ValueError):
        batch_runtime.batch_generate(
            model,
            FakeTokenizer(),
            ["over-limit"],
            max_seq_len=2,
            batch_size=1,
            prefill_step=2,
        )

    release_probe.assert_all_released()
    capsys.readouterr()


class FailingDetokenizer(FakeDetokenizer):
    def add_token(self, token):
        raise RuntimeError("injected detokenization failure")


class FailingTokenizer(FakeTokenizer):
    detokenizer = FailingDetokenizer(FakeTokenizer._tokenizer)


@pytest.mark.parametrize(
    "failure_point", ["prefill", "materialize", "decode", "detokenize"]
)
def test_failures_release_every_owned_request_cache(capsys, failure_point):
    release_probe = CacheReleaseProbe()
    model = SchedulerModel(
        release_probe=release_probe,
        fail_at=None if failure_point == "detokenize" else failure_point,
    )
    tokenizer = FailingTokenizer() if failure_point == "detokenize" else FakeTokenizer()

    with pytest.raises(RuntimeError):
        batch_runtime.batch_generate(
            model,
            tokenizer,
            ["active"],
            max_seq_len=8,
            batch_size=2,
            prefill_step=2,
        )

    release_probe.assert_all_released()
    capsys.readouterr()
