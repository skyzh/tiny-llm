import mlx.core as mx
import pytest

from extensions_ref.tiny_llm_ext_ref import _ext
from tiny_llm_ref.kv_cache import BatchingKvCache
from tiny_llm_ref.paged_kv_cache import TinyKvPagedCache, TinyKvPagedPool

SHAPE = (5, 8, 1, 128)


@pytest.fixture
def grouped_writer():
    if (
        mx.__version__ != "0.32.4"
        or not mx.metal.is_available()
        or mx.device_info()["architecture"] != "applegpu_g16s"
    ):
        pytest.skip("The grouped writer is scoped to MLX 0.32.4 and g16s Metal")
    with mx.stream(mx.gpu):
        yield _ext._try_grouped_paged_cache_update


def _words(shape, bias=0):
    size = 1
    for axis in shape:
        size *= axis
    return (mx.arange(size).reshape(shape) % 59 + bias).astype(mx.bfloat16)


def _frame(lengths=(127, None, 254, None, 63)):
    pool = TinyKvPagedPool()
    batch = BatchingKvCache(5)
    for slot, length in enumerate(lengths):
        if length is not None:
            cache = TinyKvPagedCache(pool)
            batch.add_request(cache, slot)
            if length:
                shape = (1, 8, length, 128)
                cache.update_and_fetch_paged(_words(shape, slot), _words(shape, 60))
    mx.eval(pool._key_pages, pool._value_pages)
    return batch, pool


def _inputs(pool, case):
    keys = _words(SHAPE, 100)
    values = _words(SHAPE, 160)
    if case == "late-layout":
        values = _words((8, 5, 1, 128), 160).transpose(1, 0, 2, 3)
    if case == "late-key-alias":
        keys = pool._key_pages.reshape(-1)[:5120].reshape(SHAPE)
    if case == "late-value-alias":
        values = pool._key_pages.reshape(-1)[:5120].reshape(SHAPE)
    return keys, values


def _roots(pool):
    return pool._key_pages, pool._value_pages


def _assert_exact(left, right):
    for a, b in zip(left, right, strict=True):
        assert mx.array_equal(
            mx.contiguous(a).view(mx.uint16), mx.contiguous(b).view(mx.uint16)
        ).item()


@pytest.mark.parametrize(
    "case,lengths,expected",
    [
        ("grouped", (127, None, 254, None, 63), [True]),
        ("late-layout", (127, None, 254, None, 63), [False]),
        ("late-key-alias", (127, None, 254, None, 63), [False]),
        ("late-value-alias", (127, None, 254, None, 63), [False]),
        ("lazy-storage", (127, None, 254, None, 63), [False]),
        ("one-slot", (None, None, None, 63, None), []),
        ("first-page", (127, None, 0, None, 63), []),
        ("rollover-and-growth", (128, None, 254, None, 63), []),
    ],
)
def test_grouped_tail_write_matches_stock(
    grouped_writer, monkeypatch, case, lengths, expected
):
    actual, pool = _frame(lengths)
    stock, stock_pool = _frame(lengths)
    held = pool.read_page(0)
    stock_held = stock_pool.read_page(0)
    mx.eval(*held, *stock_held)
    if case == "lazy-storage":
        pool._value_pages = pool._value_pages.reshape(-1).reshape(
            pool._value_pages.shape
        )
        stock_pool._value_pages = stock_pool._value_pages.reshape(-1).reshape(
            stock_pool._value_pages.shape
        )
    observed = []

    def observe(*args):
        roots = grouped_writer(*args)
        observed.append(roots is not None)
        return roots

    monkeypatch.setattr(_ext, "_try_grouped_paged_cache_update", observe)
    meta = actual.update_and_fetch_paged(*_inputs(pool, case))
    monkeypatch.setattr(_ext, "_try_grouped_paged_cache_update", None)
    old_meta = stock.update_and_fetch_paged(*_inputs(stock_pool, case))
    assert observed == expected
    _assert_exact(_roots(pool), _roots(stock_pool))
    _assert_exact(held, stock_held)
    assert mx.array_equal(meta.block_table, old_meta.block_table).item()
    assert mx.array_equal(meta.context_lens, old_meta.context_lens).item()
    assert pool._snapshot_state()[2:] == stock_pool._snapshot_state()[2:]

    for slot, length in enumerate(lengths):
        if length is not None:
            actual.remove_request(slot)
            stock.remove_request(slot)
    assert not pool.used_page_ids
    assert pool.free_page_ids == stock_pool.free_page_ids


def test_grouped_tail_write_rolls_back_both_roots_and_retries(
    grouped_writer, monkeypatch
):
    batch, pool = _frame()
    stock, stock_pool = _frame()
    before = pool._snapshot_state()
    caches = [cache for cache in batch.kv_caches if cache is not None]
    states = [cache._snapshot_state() for cache in caches]
    original_metadata = TinyKvPagedCache.paged_metadata

    def fail_after_writers_are_constructed(cache, *args, **kwargs):
        if cache is caches[1]:
            assert all(
                new is not original for new, original in zip(_roots(pool), before[:2])
            )
            raise RuntimeError("injected metadata failure")
        return original_metadata(cache, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            TinyKvPagedCache, "paged_metadata", fail_after_writers_are_constructed
        )
        with pytest.raises(RuntimeError, match="injected metadata failure"):
            batch.update_and_fetch_paged(*_inputs(pool, "grouped"))
    assert all(
        new is original for new, original in zip(_roots(pool), before[:2], strict=True)
    )
    assert pool._snapshot_state()[2:] == before[2:]
    assert [cache._snapshot_state() for cache in caches] == states
    assert batch.HD is None
    _assert_exact(_roots(pool), _roots(stock_pool))

    batch.update_and_fetch_paged(*_inputs(pool, "grouped"))
    monkeypatch.setattr(_ext, "_try_grouped_paged_cache_update", None)
    stock.update_and_fetch_paged(*_inputs(stock_pool, "grouped"))
    _assert_exact(_roots(pool), _roots(stock_pool))
