from types import SimpleNamespace

import mlx.core as mx
import pytest

from tiny_llm_ref._prefix_page import LAYERS, PAGE_SHAPE, PAGE_SIZE, _FirstPageCopy
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
