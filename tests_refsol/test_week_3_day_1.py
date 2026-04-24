from types import SimpleNamespace

import mlx.core as mx

from .tiny_llm_base import (
    Qwen2ModelWeek2,
    Qwen2ModelWeek3,
    TinyKvFullCache,
    TinyKvPagedCache,
    TinyKvPagedPool,
)
from .utils import assert_allclose


def _random_chunk(length: int, num_heads: int = 2, head_dim: int = 4) -> tuple[mx.array, mx.array]:
    key = mx.random.normal(shape=(1, num_heads, length, head_dim)).astype(mx.float32)
    value = mx.random.normal(shape=(1, num_heads, length, head_dim)).astype(mx.float32)
    return key, value


def _quantized_layer(
    out_dim: int, in_dim: int, *, bias: bool = False, group_size: int = 64
) -> SimpleNamespace:
    weight = mx.random.normal(shape=(out_dim, in_dim), dtype=mx.float16)
    quantized_weight, scales, biases = mx.quantize(weight, group_size=group_size, bits=4)
    layer = SimpleNamespace(
        weight=quantized_weight,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=4,
    )
    if bias:
        layer.bias = mx.random.normal(shape=(out_dim,), dtype=mx.float16)
    return layer


def _fake_qwen2_mlx_model() -> SimpleNamespace:
    mx.random.seed(0)
    args = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=64,
        vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
        rope_theta=10000,
        tie_word_embeddings=True,
    )
    embed_tokens = _quantized_layer(args.vocab_size, args.hidden_size)
    kv_hidden_size = args.hidden_size // args.num_attention_heads * args.num_key_value_heads
    layers = []
    for _ in range(args.num_hidden_layers):
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    q_proj=_quantized_layer(args.hidden_size, args.hidden_size, bias=True),
                    k_proj=_quantized_layer(kv_hidden_size, args.hidden_size, bias=True),
                    v_proj=_quantized_layer(kv_hidden_size, args.hidden_size, bias=True),
                    o_proj=_quantized_layer(args.hidden_size, args.hidden_size),
                ),
                mlp=SimpleNamespace(
                    gate_proj=_quantized_layer(args.intermediate_size, args.hidden_size),
                    up_proj=_quantized_layer(args.intermediate_size, args.hidden_size),
                    down_proj=_quantized_layer(args.hidden_size, args.intermediate_size),
                ),
                input_layernorm=SimpleNamespace(
                    weight=mx.ones((args.hidden_size,), dtype=mx.float16)
                ),
                post_attention_layernorm=SimpleNamespace(
                    weight=mx.ones((args.hidden_size,), dtype=mx.float16)
                ),
            )
        )
    return SimpleNamespace(
        args=args,
        model=SimpleNamespace(
            embed_tokens=embed_tokens,
            layers=layers,
            norm=SimpleNamespace(weight=mx.ones((args.hidden_size,), dtype=mx.float16)),
        ),
    )


def test_task_1_paged_cache_matches_full_cache():
    page_size = 4
    full = TinyKvFullCache()
    pool = TinyKvPagedPool(page_size=page_size)
    paged = TinyKvPagedCache(pool=pool)

    total_len = 0
    for length in [3, 2, 5]:
        key, value = _random_chunk(length)
        full_key, full_value, full_len, _ = full.update_and_fetch(key, value)
        paged_key, paged_value, paged_len, _ = paged.update_and_fetch(key, value)
        total_len += length
        assert full_len == paged_len == total_len
        assert paged.num_pages == (total_len + page_size - 1) // page_size
        physical_page_capacity = [
            paged.pool.read_page(page_id)[0].shape[2] for page_id in paged.page_ids
        ]
        assert physical_page_capacity == [page_size] * paged.num_pages
        assert sum(paged.page_lens) == total_len
        assert_allclose(paged_key, full_key, precision=mx.float32)
        assert_allclose(paged_value, full_value, precision=mx.float32)


def test_task_1_paged_pool_reuses_freed_pages():
    pool = TinyKvPagedPool(page_size=4)
    first = TinyKvPagedCache(pool=pool)
    second = TinyKvPagedCache(pool=pool)

    key, value = _random_chunk(6)
    first.update_and_fetch(key, value)
    assert first.page_ids == [0, 1]
    assert pool.num_pages == 2
    assert pool.num_free_pages == 0

    first.release()
    assert first.offset == 0
    assert pool.num_pages == 2
    assert pool.num_free_pages == 2

    second_key, second_value = _random_chunk(5)
    gathered_key, gathered_value, seq_len, _ = second.update_and_fetch(second_key, second_value)
    assert seq_len == 5
    assert pool.num_pages == 2
    assert pool.num_free_pages == 0
    assert set(second.page_ids) == {0, 1}
    assert_allclose(gathered_key, second_key, precision=mx.float32)
    assert_allclose(gathered_value, second_value, precision=mx.float32)


def test_task_1_paged_cache_rewind():
    page_size = 4
    pool = TinyKvPagedPool(page_size=page_size)
    paged = TinyKvPagedCache(pool=pool)
    full = TinyKvFullCache()

    for length in [4, 3, 2]:
        key, value = _random_chunk(length)
        paged.update_and_fetch(key, value)
        full.update_and_fetch(key, value)

    assert paged.page_lens == [4, 4, 1]
    paged.rewind(3)
    full.rewind(3)

    paged_key, paged_value = paged.gather_dense()
    full_key, full_value = full.key_values
    assert paged.offset == full.offset == 6
    assert paged.page_lens == [4, 2]
    assert paged.num_pages == 2
    assert paged.pool.num_pages == 3
    assert paged.pool.num_free_pages == 1
    physical_page_capacity = [
        paged.pool.read_page(page_id)[0].shape[2] for page_id in paged.page_ids
    ]
    assert physical_page_capacity == [page_size] * paged.num_pages
    assert_allclose(paged_key, full_key, precision=mx.float32)
    assert_allclose(paged_value, full_value, precision=mx.float32)


def test_task_1_model_kv_caches_share_layer_pools():
    mlx_model = _fake_qwen2_mlx_model()
    week3_model = Qwen2ModelWeek3(mlx_model, page_size=4)
    first_request_cache = week3_model.create_kv_cache()
    second_request_cache = week3_model.create_kv_cache()

    assert len(first_request_cache) == week3_model.num_hidden_layers
    for layer in range(week3_model.num_hidden_layers):
        assert first_request_cache[layer].pool is week3_model.page_pool
        assert second_request_cache[layer].pool is week3_model.page_pool

    assert first_request_cache[0].page_ids is not first_request_cache[1].page_ids
    assert first_request_cache[0].page_lens is not first_request_cache[1].page_lens
    assert first_request_cache[0].pool is first_request_cache[1].pool


def test_task_1_model_layer_caches_keep_independent_page_metadata():
    mlx_model = _fake_qwen2_mlx_model()
    week3_model = Qwen2ModelWeek3(mlx_model, page_size=4)
    cache = week3_model.create_kv_cache()
    inputs = mx.array([[1, 5, 7, 3, 9]], dtype=mx.int32)

    week3_model(inputs, 0, cache)

    assert cache[0].page_ids == [0, 1]
    assert cache[0].page_lens == [4, 1]
    owned_page_ids = set(cache[0].page_ids)
    for layer in range(1, week3_model.num_hidden_layers):
        assert cache[layer].page_lens == cache[0].page_lens
        assert owned_page_ids.isdisjoint(cache[layer].page_ids)
        owned_page_ids.update(cache[layer].page_ids)
        for page_id in cache[layer].page_ids:
            key_page, value_page = week3_model.page_pool.read_page(page_id)
            assert key_page.shape[2] == week3_model.page_size
            assert value_page.shape[2] == week3_model.page_size


def test_task_3_incremental_decode_matches_week2():
    mlx_model = _fake_qwen2_mlx_model()
    week2_model = Qwen2ModelWeek2(mlx_model)
    week3_model = Qwen2ModelWeek3(mlx_model, page_size=4)
    inputs = mx.array([[1, 5, 7, 3, 9, 11]], dtype=mx.int32)
    week2_cache = week2_model.create_kv_cache()
    week3_cache = week3_model.create_kv_cache()

    for offset in range(inputs.shape[1]):
        token = inputs[:, offset : offset + 1]
        week2_out = week2_model(token, offset, week2_cache)
        week3_out = week3_model(token, offset, week3_cache)
        week2_out = week2_out - mx.logsumexp(week2_out, keepdims=True)
        week3_out = week3_out - mx.logsumexp(week3_out, keepdims=True)
        assert_allclose(week3_out, week2_out, precision=mx.float16, rtol=1e-3, atol=1e-3)
