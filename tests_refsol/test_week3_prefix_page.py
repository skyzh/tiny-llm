from pathlib import Path
import random
import runpy
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx_lm
import pytest

from tiny_llm_ref import batch, models
from tiny_llm_ref._prefix_page import LAYERS, PAGE_SHAPE, PAGE_SIZE, _FirstPageCopy
from tiny_llm_ref.kv_cache import BatchingKvCache
from tiny_llm_ref.paged_kv_cache import TinyKvPagedPool
from tiny_llm_ref.qwen3_week3 import Qwen3ModelWeek3


@pytest.fixture
def model():
    model = Qwen3ModelWeek3.__new__(Qwen3ModelWeek3)
    model.page_size = PAGE_SIZE
    model.precision = mx.bfloat16
    model.num_hidden_layers = LAYERS
    model.page_pools = [TinyKvPagedPool(page_size=PAGE_SIZE) for _ in range(LAYERS)]
    attn = SimpleNamespace(use_paged_attention=True, num_kv_heads=8, head_dim=128)
    model.layers_inner = [SimpleNamespace(self_attn=attn) for _ in range(LAYERS)]
    return model


def request(model, first_token=0, suffix=7, page_value=1):
    tokens = [first_token] + [0] * (PAGE_SIZE - 1) + [suffix]
    pending = SimpleNamespace(
        model=model,
        offset=0,
        is_prefill_done=False,
        prefill_max_step=PAGE_SIZE,
        prefill_tokens=mx.array(tokens, dtype=mx.int32),
        kv_cache=model.create_kv_cache(),
    )
    page = mx.full(PAGE_SHAPE, page_value, dtype=mx.bfloat16)

    def prefill():
        for cache in pending.kv_cache:
            cache._append_chunk(page, -page)
            cache.materialize()
        pending.offset += PAGE_SIZE

    pending.try_prefill = prefill
    return pending


def release(pending):
    for cache in pending.kv_cache:
        cache.release()


def test_reuse_requires_a_full_matching_token_page_and_detaches_it(model):
    store = _FirstPageCopy.for_queue(model, PAGE_SIZE)
    assert store is not None
    donor = request(model)
    hit = request(model, suffix=9, page_value=2)
    miss = request(model, first_token=1, page_value=3)

    try:
        store.prefill(donor)
        store.prefill(hit)
        assert store.captures == store.hits == 1
        assert hit.offset == PAGE_SIZE and not hit.is_prefill_done
        assert all(
            a.page_ids[0] != b.page_ids[0]
            for a, b in zip(donor.kv_cache, hit.kv_cache, strict=True)
        )

        donor_ids = [c.page_ids[0] for c in donor.kv_cache]
        release(donor)
        store.prefill(miss)
        assert store.captures == store.hits == 1
        assert [c.page_ids[0] for c in miss.kv_cache] == donor_ids
        for cache, (key, value) in zip(
            hit.kv_cache, store.snapshot.layers, strict=True
        ):
            actual_key, actual_value = cache.gather_dense()
            assert mx.all(key == 1).item() and mx.all(value == -1).item()
            assert mx.array_equal(actual_key, key).item()
            assert mx.array_equal(actual_value, value).item()
    finally:
        for pending in (donor, hit, miss):
            release(pending)
        store.close()
    assert all(not pool.used_page_ids for pool in model.page_pools)


def test_failed_prefix_copy_restores_all_layers_before_retry(model, monkeypatch):
    store = _FirstPageCopy.for_queue(model, PAGE_SIZE)
    assert store is not None
    donor, hit = request(model), request(model, suffix=9, page_value=2)

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected page write failure")

    try:
        store.prefill(donor)
        snapshot = store.snapshot
        states = [pool._snapshot_state() for pool in model.page_pools]
        with monkeypatch.context() as patch:
            patch.setattr(model.page_pools[1], "write_page_slice", fail)
            with pytest.raises(RuntimeError, match="injected page write failure"):
                store.prefill(hit)
        assert store.snapshot is snapshot and store.hits == hit.offset == 0
        assert all(cache.offset == 0 and not cache.page_ids for cache in hit.kv_cache)
        for previous, pool in zip(states, model.page_pools, strict=True):
            current = pool._snapshot_state()
            assert current[0] is previous[0] and current[1] is previous[1]
            assert current[2:] == previous[2:]
        store.prefill(hit)
        assert store.hits == 1 and hit.offset == PAGE_SIZE
    finally:
        release(donor)
        release(hit)
        store.close()
    assert all(not pool.used_page_ids for pool in model.page_pools)


@pytest.mark.parametrize("case", ["hit", "miss", "custom", "fallback", "failure"])
def test_final_prefix_lookup_through_cli(model, monkeypatch, case):
    recipient_key = 5 if case == "miss" else 3
    if case == "fallback":
        model.precision = mx.float32

    class Detokenizer:
        def __init__(self, _tokenizer):
            self.text = ""

        def add_token(self, token):
            self.text += str(token)

    tokens = {
        "donor": [3] * PAGE_SIZE + [7],
        "recipient": [recipient_key] * PAGE_SIZE + [9],
    }
    tokenizer = SimpleNamespace(
        _tokenizer=None,
        detokenizer=Detokenizer(None),
        eos_token_id=0,
        encode=lambda prompt, **_options: tokens[prompt],
        apply_chat_template=lambda messages, **_options: messages[-1]["content"],
    )

    def step(_model, inputs, offsets, caches):
        if isinstance(caches[0], BatchingKvCache):
            return mx.zeros((len(offsets),), dtype=mx.int32)
        token = inputs[0, 0].item()
        shape = (*PAGE_SHAPE[:2], inputs.shape[1], PAGE_SHAPE[-1])
        key = mx.full(shape, token, dtype=mx.bfloat16)
        for cache in caches:
            cache._append_chunk(key, -key)
        return mx.array([token])

    prefill, close = _FirstPageCopy.prefill, _FirstPageCopy.close
    store_at_final = None
    paid, after, closed = [], [], []

    def fail_write(*_args, **_kwargs):
        raise RuntimeError("final page write failed")

    def track_prefill(store, request):
        nonlocal store_at_final
        if request.prompt_idx == 0:
            prefill(store, request)
            return
        store_at_final = store
        assert store.snapshot is not None
        with monkeypatch.context() as patch:
            if case == "failure":
                patch.setattr(model.page_pools[1], "write_page_slice", fail_write)
            prefill(store, request)
        paid.append((store.snapshot is not None, request.offset, store.hits))

    def track_progress(_slots, pending, _size, _count, _start):
        if pending is None or pending.prompt_idx != 1 or pending.offset != PAGE_SIZE:
            return
        for cache, pool in zip(pending.kv_cache, model.page_pools, strict=True):
            assert cache.pool is pool and set(cache.page_ids) == pool.used_page_ids
            key, value = cache.gather_dense()
            assert mx.all(key == recipient_key).item()
            assert mx.all(value == -recipient_key).item()
        after.append(store_at_final is None or store_at_final.snapshot is None)

    def track_close(store):
        closed.append(store.snapshot is None)
        close(store)
        assert store.snapshot is None

    monkeypatch.setattr(batch, "_step", step)
    monkeypatch.setattr(batch, "_print_progress", track_progress)
    monkeypatch.setattr(_FirstPageCopy, "prefill", track_prefill)
    monkeypatch.setattr(_FirstPageCopy, "close", track_close)
    cli = Path(__file__).resolve().parents[1] / "batch-main.py"
    monkeypatch.setattr(mlx_lm, "load", lambda _name: (object(), tokenizer))
    monkeypatch.setattr(models, "dispatch_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        random, "shuffle", lambda prompts: prompts.__setitem__(slice(None), tokens)
    )
    monkeypatch.setattr(
        sys, "argv", [str(cli), "--solution", "ref", "--loader", "week3"]
    )

    if case == "failure":
        with pytest.raises(RuntimeError, match="final page write failed"):
            runpy.run_path(str(cli), run_name="__main__")
        assert paid == after == []
        assert closed == [False]
    else:
        if case == "custom":
            result = batch._batch_generate_with_cli_tokenizer(
                model,
                tokenizer,
                tokens,
                max_seq_len=512,
                batch_size=5,
                prefill_step=128,
            )
        else:
            result = runpy.run_path(str(cli), run_name="__main__")["result"]
        assert result == [(0, "7"), (1, "9")]
        if case == "fallback":
            assert paid == closed == []
            assert after == [True]
        else:
            assert paid == [(True, PAGE_SIZE, int(case != "miss"))]
            assert after == closed == [case != "custom"]
    assert all(not pool.used_page_ids for pool in model.page_pools)
