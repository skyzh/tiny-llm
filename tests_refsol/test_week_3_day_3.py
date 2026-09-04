"""Week 3 Day 3 paged-KV storage tests."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

if __package__ == "tests_refsol":
    from extensions_ref import tiny_llm_ext_ref as paged_extension
else:
    from extensions import tiny_llm_ext as paged_extension

from .tiny_llm_base import (
    BatchingKvCache,
    Qwen3ModelWeek2,
    Qwen3ModelWeek3,
    TinyKvFullCache,
    TinyKvPagedCache,
    TinyKvPagedPool,
)
from .utils import assert_allclose


def _random_chunk(
    length: int,
    num_heads: int = 2,
    head_dim: int = 4,
    dtype: mx.Dtype = mx.float32,
) -> tuple[mx.array, mx.array]:
    key = mx.random.normal(shape=(1, num_heads, length, head_dim)).astype(dtype)
    value = mx.random.normal(shape=(1, num_heads, length, head_dim)).astype(dtype)
    return key, value


def _logical_contents(cache: TinyKvPagedCache) -> tuple[mx.array, mx.array] | None:
    if cache.offset == 0:
        return None
    key, value = cache.gather_dense()
    mx.eval(key, value)
    return key, value


def _assert_logical_contents(
    cache: TinyKvPagedCache,
    expected: tuple[mx.array, mx.array] | None,
) -> None:
    actual = _logical_contents(cache)
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    assert_allclose(actual[0], expected[0], precision=mx.float32)
    assert_allclose(actual[1], expected[1], precision=mx.float32)


def _quantized_layer(
    out_dim: int, in_dim: int, group_size: int = 128
) -> SimpleNamespace:
    weight = mx.random.normal(shape=(out_dim, in_dim), dtype=mx.bfloat16)
    quantized_weight, scales, biases = mx.quantize(
        weight, group_size=group_size, bits=4
    )
    return SimpleNamespace(
        weight=quantized_weight,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=4,
    )


def _fake_qwen3_mlx_model(
    tie_word_embeddings: bool = True,
    seed: int = 0,
) -> SimpleNamespace:
    mx.random.seed(seed)
    args = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=128,
        vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
        rope_theta=10000,
        tie_word_embeddings=tie_word_embeddings,
    )
    embed_tokens = _quantized_layer(args.vocab_size, args.hidden_size)
    kv_hidden_size = args.num_key_value_heads * args.head_dim
    attn_hidden_size = args.num_attention_heads * args.head_dim
    layers = []
    for _ in range(args.num_hidden_layers):
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    q_proj=_quantized_layer(attn_hidden_size, args.hidden_size),
                    k_proj=_quantized_layer(kv_hidden_size, args.hidden_size),
                    v_proj=_quantized_layer(kv_hidden_size, args.hidden_size),
                    o_proj=_quantized_layer(args.hidden_size, attn_hidden_size),
                    q_norm=SimpleNamespace(
                        weight=mx.ones((args.head_dim,), dtype=mx.bfloat16)
                    ),
                    k_norm=SimpleNamespace(
                        weight=mx.ones((args.head_dim,), dtype=mx.bfloat16)
                    ),
                ),
                mlp=SimpleNamespace(
                    gate_proj=_quantized_layer(
                        args.intermediate_size, args.hidden_size
                    ),
                    up_proj=_quantized_layer(args.intermediate_size, args.hidden_size),
                    down_proj=_quantized_layer(
                        args.hidden_size, args.intermediate_size
                    ),
                ),
                input_layernorm=SimpleNamespace(
                    weight=mx.ones((args.hidden_size,), dtype=mx.bfloat16)
                ),
                post_attention_layernorm=SimpleNamespace(
                    weight=mx.ones((args.hidden_size,), dtype=mx.bfloat16)
                ),
            )
        )
    return SimpleNamespace(
        args=args,
        model=SimpleNamespace(
            embed_tokens=embed_tokens,
            layers=layers,
            norm=SimpleNamespace(
                weight=mx.ones((args.hidden_size,), dtype=mx.bfloat16)
            ),
        ),
        lm_head=_quantized_layer(args.vocab_size, args.hidden_size),
    )


def test_task_1_paged_cache_matches_full_cache():
    page_size = 4
    full = TinyKvFullCache()
    paged = TinyKvPagedCache(pool=TinyKvPagedPool(page_size=page_size))

    total_len = 0
    for length in [3, 2, 5]:
        key, value = _random_chunk(length)
        full_key, full_value, full_len, _ = full.update_and_fetch(key, value)
        paged_key, paged_value, paged_len, _ = paged.update_and_fetch(key, value)
        total_len += length
        assert full_len == paged_len == total_len
        assert paged.num_pages == (total_len + page_size - 1) // page_size
        assert sum(paged.page_lens) == total_len
        assert_allclose(paged_key, full_key, precision=mx.float32)
        assert_allclose(paged_value, full_value, precision=mx.float32)


def test_task_1_paged_pool_reuses_freed_capacity():
    pool = TinyKvPagedPool(page_size=4)
    first = TinyKvPagedCache(pool=pool)
    second = TinyKvPagedCache(pool=pool)

    first.update_and_fetch(*_random_chunk(6))
    original_pages = pool.num_pages
    original_capacity = pool.capacity
    first.release()

    key, value = _random_chunk(5)
    gathered_key, gathered_value, seq_len, _ = second.update_and_fetch(key, value)
    assert seq_len == 5
    assert pool.num_pages == original_pages
    assert pool.capacity == original_capacity
    assert pool.num_free_pages == 0
    assert_allclose(gathered_key, key, precision=mx.float32)
    assert_allclose(gathered_value, value, precision=mx.float32)


def test_task_1_rejects_incompatible_dtype_without_mutating_cache():
    cache = TinyKvPagedCache(pool=TinyKvPagedPool(page_size=4))
    cache.update_and_fetch_paged(*_random_chunk(4))
    before = _logical_contents(cache)
    before_shape = (cache.offset, cache.num_pages, cache.pool.capacity)
    key, value = _random_chunk(1, dtype=mx.bfloat16)

    with pytest.raises(ValueError):
        cache.update_and_fetch_paged(key, value)

    assert (cache.offset, cache.num_pages, cache.pool.capacity) == before_shape
    _assert_logical_contents(cache, before)


def test_task_1_rejects_shape_mismatch_without_mutating_cache():
    cache = TinyKvPagedCache(pool=TinyKvPagedPool(page_size=4))
    before_shape = (cache.offset, cache.num_pages, cache.pool.capacity)
    key, value = _random_chunk(2)

    with pytest.raises(ValueError):
        cache.update_and_fetch_paged(key, value[:, :, :1, :])

    assert (cache.offset, cache.num_pages, cache.pool.capacity) == before_shape
    _assert_logical_contents(cache, None)


def test_task_1_rolls_back_a_multi_page_append(monkeypatch):
    cache = TinyKvPagedCache(pool=TinyKvPagedPool(page_size=4))
    cache.update_and_fetch_paged(*_random_chunk(2))
    before = _logical_contents(cache)
    before_shape = (
        cache.offset,
        cache.num_pages,
        cache.pool.num_pages,
        cache.pool.num_free_pages,
        cache.pool.capacity,
    )
    original_write = cache.pool.write_page_slice
    writes = 0

    def fail_after_one_page(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes > 1:
            raise RuntimeError("injected failure")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(cache.pool, "write_page_slice", fail_after_one_page)
    with pytest.raises(RuntimeError):
        cache.update_and_fetch_paged(*_random_chunk(7))

    assert (
        cache.offset,
        cache.num_pages,
        cache.pool.num_pages,
        cache.pool.num_free_pages,
        cache.pool.capacity,
    ) == before_shape
    _assert_logical_contents(cache, before)


def test_task_1_mixed_pools_fail_before_any_batch_row_mutates():
    first = TinyKvPagedCache(pool=TinyKvPagedPool(page_size=4))
    second = TinyKvPagedCache(pool=TinyKvPagedPool(page_size=4))
    batch = BatchingKvCache(max_active_requests=2, max_seq_len=8)
    batch.add_request(first, 0)
    batch.add_request(second, 1)
    keys = mx.zeros((2, 2, 1, 4), dtype=mx.float32)

    with pytest.raises(ValueError):
        batch.update_and_fetch_paged(keys, keys, mask_length=1)

    assert first.offset == second.offset == 0
    assert first.pool.num_pages == second.pool.num_pages == 0


def test_task_1_batch_append_is_transactional(monkeypatch):
    pool = TinyKvPagedPool(page_size=4)
    first = TinyKvPagedCache(pool=pool)
    second = TinyKvPagedCache(pool=pool)
    first.update_and_fetch_paged(*_random_chunk(2))
    second.update_and_fetch_paged(*_random_chunk(3))
    first_before = _logical_contents(first)
    second_before = _logical_contents(second)
    before_shape = (
        first.offset,
        second.offset,
        pool.num_pages,
        pool.num_free_pages,
        pool.capacity,
    )
    batch = BatchingKvCache(max_active_requests=2, max_seq_len=16)
    batch.add_request(first, 0)
    batch.add_request(second, 1)

    def reject_append(*args, **kwargs):
        raise RuntimeError("injected request failure")

    monkeypatch.setattr(second, "update_and_fetch_paged", reject_append)
    keys = mx.random.normal(shape=(2, 2, 3, 4)).astype(mx.float32)
    with pytest.raises(RuntimeError):
        batch.update_and_fetch_paged(keys, keys, mask_length=3)

    assert (
        first.offset,
        second.offset,
        pool.num_pages,
        pool.num_free_pages,
        pool.capacity,
    ) == before_shape
    _assert_logical_contents(first, first_before)
    _assert_logical_contents(second, second_before)


def test_task_1_paged_pool_grows_geometrically_and_preserves_data():
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvPagedCache(pool=pool)
    expected_keys = []
    expected_values = []
    capacities = []

    for _ in range(10):
        key, value = _random_chunk(4)
        expected_keys.append(key)
        expected_values.append(value)
        cache.update_and_fetch_paged(key, value)
        capacities.append(pool.capacity)

    changed_capacities = [
        capacity
        for index, capacity in enumerate(capacities)
        if index == 0 or capacity != capacities[index - 1]
    ]
    assert all(
        current >= previous * 2
        for previous, current in zip(changed_capacities, changed_capacities[1:])
    )
    assert len(changed_capacities) < cache.num_pages
    assert pool.capacity >= pool.num_pages
    assert pool.capacity < pool.num_pages * 2
    gathered_key, gathered_value = cache.gather_dense()
    assert_allclose(
        gathered_key, mx.concat(expected_keys, axis=2), precision=mx.float32
    )
    assert_allclose(
        gathered_value, mx.concat(expected_values, axis=2), precision=mx.float32
    )


def test_task_1_paged_pool_reset_removes_warmup_capacity():
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvPagedCache(pool=pool)
    cache.update_and_fetch_paged(*_random_chunk(17))
    cache.release()

    assert pool.capacity > 0
    assert pool.num_free_pages == pool.num_pages
    pool.reset()

    assert pool.capacity == 0
    assert pool.num_pages == 0
    assert pool.num_free_pages == 0
    assert pool.storage_nbytes == 0


def test_task_1_block_table_tracks_logical_pages():
    cache = TinyKvPagedCache(pool=TinyKvPagedPool(page_size=4))
    cache.update_and_fetch_paged(*_random_chunk(3))
    initial = cache.block_table().tolist()

    cache.update_and_fetch_paged(*_random_chunk(1))
    assert cache.block_table().tolist() == initial

    cache.update_and_fetch_paged(*_random_chunk(1))
    expanded = cache.block_table().tolist()
    assert expanded[0][:-1] == initial[0]
    assert len(expanded[0]) == len(initial[0]) + 1


def test_task_1_materialize_keeps_logical_contents_available():
    cache = TinyKvPagedCache(pool=TinyKvPagedPool(page_size=4))
    key, value = _random_chunk(5)
    cache.update_and_fetch_paged(key, value)

    cache.materialize()
    gathered_key, gathered_value = cache.gather_dense()
    mx.eval(gathered_key)
    mx.eval(gathered_value)

    assert_allclose(gathered_key, key, precision=mx.float32)
    assert_allclose(gathered_value, value, precision=mx.float32)


def test_task_1_paged_cache_rewind_matches_full_cache():
    paged = TinyKvPagedCache(pool=TinyKvPagedPool(page_size=4))
    full = TinyKvFullCache()

    for length in [4, 3, 2]:
        key, value = _random_chunk(length)
        paged.update_and_fetch(key, value)
        full.update_and_fetch(key, value)

    paged.rewind(3)
    full.rewind(3)

    paged_key, paged_value = paged.gather_dense()
    full_key, full_value = full.key_values
    assert paged.offset == full.offset == 6
    assert paged.num_pages == 2
    assert paged.pool.num_free_pages == 1
    assert_allclose(paged_key, full_key, precision=mx.float32)
    assert_allclose(paged_value, full_value, precision=mx.float32)


def test_task_1_paged_cache_rewind_then_append_matches_full_cache():
    paged = TinyKvPagedCache(pool=TinyKvPagedPool(page_size=4))
    full = TinyKvFullCache()
    prefix = _random_chunk(9)
    paged.update_and_fetch(*prefix)
    full.update_and_fetch(*prefix)
    paged.rewind(5)
    full.rewind(5)
    suffix = _random_chunk(6)

    paged_key, paged_value, paged_len, _ = paged.update_and_fetch(*suffix)
    full_key, full_value, full_len, _ = full.update_and_fetch(*suffix)

    assert paged_len == full_len == 10
    assert_allclose(paged_key, full_key, precision=mx.float32)
    assert_allclose(paged_value, full_value, precision=mx.float32)


def test_task_1_request_caches_keep_independent_metadata():
    pool = TinyKvPagedPool(page_size=4)
    first = TinyKvPagedCache(pool=pool)
    second = TinyKvPagedCache(pool=pool)
    first_key, first_value = _random_chunk(5)
    first.update_and_fetch_paged(first_key, first_value)

    assert second.offset == 0
    assert second.num_pages == 0
    assert second.key_values is None

    second_key, second_value = _random_chunk(3)
    second.update_and_fetch_paged(second_key, second_value)
    _assert_logical_contents(first, (first_key, first_value))
    _assert_logical_contents(second, (second_key, second_value))


def test_task_2_batch_removal_releases_capacity_for_next_request():
    pool = TinyKvPagedPool(page_size=4)
    first = TinyKvPagedCache(pool=pool)
    batch = BatchingKvCache(max_active_requests=1, max_seq_len=16)
    batch.add_request(first, 0)
    keys = mx.random.normal(shape=(1, 2, 6, 4)).astype(mx.float32)
    values = mx.random.normal(shape=(1, 2, 6, 4)).astype(mx.float32)
    batch.update_and_fetch_paged(keys, values, mask_length=6)
    pages_before = pool.num_pages
    capacity_before = pool.capacity

    batch.remove_request(0)
    assert first.offset == 0
    assert pool.num_free_pages == pages_before

    second = TinyKvPagedCache(pool=pool)
    batch.add_request(second, 0)
    next_keys = mx.random.normal(shape=(1, 2, 5, 4)).astype(mx.float32)
    next_values = mx.random.normal(shape=(1, 2, 5, 4)).astype(mx.float32)
    batch.update_and_fetch_paged(next_keys, next_values, mask_length=5)

    assert pool.num_pages == pages_before
    assert pool.capacity == capacity_before
    _assert_logical_contents(second, (next_keys, next_values))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_task_2_paged_cache_update_writes_only_the_requested_slice(dtype):
    pages = mx.zeros((3, 2, 4, 3), dtype=dtype)
    values = mx.arange(12).reshape(1, 2, 2, 3).astype(dtype)

    updated = paged_extension.paged_cache_update(pages, values, 1, 1)
    mx.eval(updated)
    expected = mx.zeros((3, 2, 4, 3), dtype=dtype)
    expected[1:2, :, 1:3, :] = values

    assert updated.shape == pages.shape
    assert updated.dtype == dtype
    assert_allclose(updated, expected, precision=mx.float32)


@pytest.mark.parametrize(
    "pages,values,page_id,start",
    [
        (
            mx.zeros((2, 2, 4), dtype=mx.float32),
            mx.zeros((1, 2, 1, 3), dtype=mx.float32),
            0,
            0,
        ),
        (
            mx.zeros((2, 2, 4, 3), dtype=mx.float32),
            mx.zeros((2, 2, 1, 3), dtype=mx.float32),
            0,
            0,
        ),
        (
            mx.zeros((2, 2, 4, 3), dtype=mx.float32),
            mx.zeros((1, 3, 1, 3), dtype=mx.float32),
            0,
            0,
        ),
        (
            mx.zeros((2, 2, 4, 3), dtype=mx.float32),
            mx.zeros((1, 2, 1, 3), dtype=mx.bfloat16),
            0,
            0,
        ),
        (
            mx.zeros((2, 2, 4, 3), dtype=mx.float32),
            mx.zeros((1, 2, 1, 3), dtype=mx.float32),
            -1,
            0,
        ),
        (
            mx.zeros((2, 2, 4, 3), dtype=mx.float32),
            mx.zeros((1, 2, 1, 3), dtype=mx.float32),
            2,
            0,
        ),
        (
            mx.zeros((2, 2, 4, 3), dtype=mx.float32),
            mx.zeros((1, 2, 2, 3), dtype=mx.float32),
            0,
            3,
        ),
    ],
)
def test_task_2_paged_cache_update_rejects_invalid_inputs(
    pages: mx.array,
    values: mx.array,
    page_id: int,
    start: int,
):
    with pytest.raises((RuntimeError, ValueError)):
        paged_extension.paged_cache_update(pages, values, page_id, start)


def test_task_3_model_request_caches_keep_independent_metadata():
    model = Qwen3ModelWeek3(
        _fake_qwen3_mlx_model(), page_size=4, enable_paged_attention=False
    )
    first = model.create_kv_cache()
    second = model.create_kv_cache()
    first_key, first_value = _random_chunk(5)
    first[0].update_and_fetch_paged(first_key, first_value)

    assert second[0].offset == 0
    assert second[0].num_pages == 0

    second_key, second_value = _random_chunk(3)
    second[0].update_and_fetch_paged(second_key, second_value)
    _assert_logical_contents(first[0], (first_key, first_value))
    _assert_logical_contents(second[0], (second_key, second_value))
    assert first[1].offset == second[1].offset == 0


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_task_3_week3_full_prompt_matches_week2(tie_word_embeddings: bool):
    inputs = mx.array([[1, 5, 7, 3, 9, 11]], dtype=mx.int32)

    for seed in (0, 11, 15):
        mlx_model = _fake_qwen3_mlx_model(
            tie_word_embeddings=tie_word_embeddings,
            seed=seed,
        )
        week2_model = Qwen3ModelWeek2(mlx_model)
        week3_model = Qwen3ModelWeek3(
            mlx_model,
            page_size=4,
            enable_paged_attention=False,
        )
        week2_out = week2_model(inputs, 0, week2_model.create_kv_cache())
        week3_out = week3_model(inputs, 0, week3_model.create_kv_cache())
        week2_out = week2_out - mx.logsumexp(week2_out, keepdims=True)
        week3_out = week3_out - mx.logsumexp(week3_out, keepdims=True)

        assert_allclose(
            week3_out,
            week2_out,
            precision=mx.bfloat16,
            rtol=1e-7,
            atol=4.0,
            message=f"seed={seed}, tie_word_embeddings={tie_word_embeddings}",
        )


def test_task_3_incremental_decode_attention_cache_matches_week2():
    mlx_model = _fake_qwen3_mlx_model()
    week2_model = Qwen3ModelWeek2(mlx_model)
    week3_model = Qwen3ModelWeek3(mlx_model, page_size=4, enable_paged_attention=False)
    inputs = mx.array([[1, 5, 7, 3, 9, 11]], dtype=mx.int32)
    week2_cache = week2_model.create_kv_cache()
    week3_cache = week3_model.create_kv_cache()

    for offset in range(inputs.shape[1]):
        token = inputs[:, offset : offset + 1]
        week2_out = week2_model(token, offset, week2_cache)
        week3_out = week3_model(token, offset, week3_cache)
        week2_out = week2_out - mx.logsumexp(week2_out, keepdims=True)
        week3_out = week3_out - mx.logsumexp(week3_out, keepdims=True)
        assert_allclose(
            week3_out, week2_out, precision=mx.bfloat16, rtol=5e-2, atol=2.0
        )
