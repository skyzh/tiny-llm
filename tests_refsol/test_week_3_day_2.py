"""Week 3 Day 2 chunked-prefill tests."""

import mlx.core as mx
import pytest

from .tiny_llm_base import (
    BatchingKvCache,
    Request,
    TinyKvFullCache,
    TinyKvPagedCache,
    TinyKvPagedPool,
    batch_generate,
)


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
        return list(range(1, len(prompt) + 1))


class FakeModel:
    num_hidden_layers = 1

    def __init__(self):
        self.calls = []

    def create_kv_cache(self):
        return [TinyKvFullCache()]

    def __call__(self, inputs, offsets, cache, logits_to_keep=1):
        offset = offsets[0] if isinstance(offsets, list) else int(offsets)
        self.calls.append((offset, inputs.shape[1]))
        length = inputs.shape[1]
        key = mx.zeros((1, 1, length, 1), dtype=mx.float32)
        cache[0].update_and_fetch(key, key)
        logits = mx.zeros((1, 1, 4), dtype=mx.float32)
        return logits.at[..., 1].add(1)


def test_chunked_prefill_bounds_work_and_advances_cache():
    model = FakeModel()
    request = Request(model, FakeTokenizer(), "1234567", prefill_max_step=3)

    request.try_prefill()
    assert request.offset == 3
    assert request.kv_cache[0].offset == 3
    assert not request.is_prefill_done

    request.try_prefill()
    assert request.offset == 6
    assert request.kv_cache[0].offset == 6
    assert not request.is_prefill_done

    request.try_prefill()
    assert request.offset == 7
    assert request.kv_cache[0].offset == 7
    assert request.is_prefill_done
    assert request.next_token == 1
    assert model.calls == [(0, 3), (3, 3), (6, 1)]

    with pytest.raises(ValueError, match="after done"):
        request.try_prefill()


class FailingMaterializePagedCache(TinyKvPagedCache):
    def materialize(self):
        super().materialize()
        raise RuntimeError("injected materialization failure")


class PagedFakeModel:
    num_hidden_layers = 1

    def __init__(self, output_token=1, fail_at=None):
        self.pool = TinyKvPagedPool(page_size=4)
        self.output_token = output_token
        self.fail_at = fail_at
        self.calls = []
        self.cache_creations = 0

    def create_kv_cache(self):
        self.cache_creations += 1
        cache_type = (
            FailingMaterializePagedCache
            if self.fail_at == "materialize"
            else TinyKvPagedCache
        )
        return [cache_type(self.pool)]

    def __call__(self, inputs, offsets, cache, logits_to_keep=1):
        offset = offsets[0] if isinstance(offsets, list) else int(offsets)
        call_number = len(self.calls) + 1
        self.calls.append((offset, inputs.shape[1]))
        key = mx.zeros((inputs.shape[0], 1, inputs.shape[1], 1), dtype=mx.float32)
        if isinstance(cache[0], BatchingKvCache):
            cache[0].update_and_fetch_paged(key, key, mask_length=inputs.shape[1])
        else:
            cache[0].update_and_fetch_paged(key, key)
        if self.fail_at == "prefill" and call_number == 1:
            raise RuntimeError("injected prefill failure")
        if self.fail_at == "decode" and call_number == 2:
            raise RuntimeError("injected decode failure")
        logits = mx.zeros((inputs.shape[0], 1, 128), dtype=mx.float32)
        return logits.at[..., self.output_token].add(1)


class FailingTextDetokenizer:
    def __init__(self, _):
        self._text = ""

    def add_token(self, token):
        self._text += str(token)

    @property
    def text(self):
        raise RuntimeError("injected detokenization failure")


class FailingTextTokenizer(FakeTokenizer):
    detokenizer = FailingTextDetokenizer(FakeTokenizer._tokenizer)


def test_request_uses_the_model_cache_factory():
    model = FakeModel()
    sentinel_cache = [TinyKvFullCache()]
    model.create_kv_cache = lambda: sentinel_cache

    request = Request(model, FakeTokenizer(), "1")

    assert request.kv_cache is sentinel_cache


def test_batch_generate_finishes_a_lone_multi_chunk_prefill():
    model = PagedFakeModel()

    result = batch_generate(
        model,
        FakeTokenizer(),
        ["1234567"],
        max_seq_len=9,
        batch_size=1,
        prefill_step=3,
    )

    assert result == [(0, "11")]
    assert model.calls == [(0, 3), (3, 3), (6, 1), (7, 1)]
    assert model.pool.used_page_ids == set()
    assert model.pool.num_free_pages == model.pool.num_pages


def test_batch_generate_finishes_prefill_eos_without_decode():
    model = PagedFakeModel(output_token=FakeTokenizer.eos_token_id)

    result = batch_generate(
        model,
        FakeTokenizer(),
        ["12345"],
        max_seq_len=10,
        batch_size=1,
        prefill_step=10,
    )

    assert result == [(0, "")]
    assert model.calls == [(0, 5)]
    assert model.pool.used_page_ids == set()
    assert model.pool.num_free_pages == model.pool.num_pages


@pytest.mark.parametrize(
    ("prompt", "expected_result", "expected_calls", "expected_creations"),
    [
        ("12", [(0, "1")], [(0, 2)], 1),
        ("123", [(0, "")], [(0, 3)], 1),
        ("1234", None, [], 0),
    ],
)
def test_batch_generate_enforces_max_seq_len_before_emission_or_allocation(
    prompt, expected_result, expected_calls, expected_creations
):
    model = PagedFakeModel()

    if expected_result is None:
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            batch_generate(model, FakeTokenizer(), [prompt], max_seq_len=3)
    else:
        assert (
            batch_generate(
                model, FakeTokenizer(), [prompt], max_seq_len=3, batch_size=1
            )
            == expected_result
        )

    assert model.calls == expected_calls
    assert model.cache_creations == expected_creations
    assert model.pool.used_page_ids == set()


@pytest.mark.parametrize(
    ("failure_point", "tokenizer"),
    [
        ("prefill", FakeTokenizer()),
        ("materialize", FakeTokenizer()),
        ("decode", FakeTokenizer()),
        ("detokenize", FailingTextTokenizer()),
    ],
)
def test_batch_generate_releases_all_paged_caches_on_exception(
    failure_point, tokenizer
):
    model = PagedFakeModel(fail_at=failure_point)

    with pytest.raises(RuntimeError, match="injected"):
        batch_generate(
            model,
            tokenizer,
            ["1"],
            max_seq_len=4,
            batch_size=1,
            prefill_step=4,
        )

    assert model.pool.used_page_ids == set()
    assert model.pool.num_free_pages == model.pool.num_pages
