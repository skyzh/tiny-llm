from types import SimpleNamespace

import mlx.core as mx

from .tiny_llm_base import (
    BatchingKvCache,
    Qwen3ModelWeek2,
    Qwen3ModelWeek3,
    TinyKvPagedCache,
    TinyKvPagedPool,
    flash_attention,
    paged_attention,
    scaled_dot_product_attention_grouped,
)
from .utils import assert_allclose


def _random_chunk(
    length: int, num_heads: int = 2, head_dim: int = 4
) -> tuple[mx.array, mx.array]:
    key = mx.random.normal(shape=(1, num_heads, length, head_dim)).astype(mx.float32)
    value = mx.random.normal(shape=(1, num_heads, length, head_dim)).astype(mx.float32)
    return key, value


def _quantized_layer(
    out_dim: int, in_dim: int, group_size: int = 64
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


def _fake_qwen3_mlx_model() -> SimpleNamespace:
    mx.random.seed(0)
    args = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=64,
        vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
        rope_theta=10000,
        tie_word_embeddings=True,
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
    )


def test_task_1_paged_attention_matches_dense_flash_attention():
    page_size = 4
    pool = TinyKvPagedPool(page_size=page_size)
    cache = TinyKvPagedCache(pool=pool)
    first_key, first_value = _random_chunk(3)
    second_key, second_value = _random_chunk(3)

    cache.update_and_fetch(first_key, first_value)
    metadata = cache.update_and_fetch_paged(second_key, second_value, mask="causal")

    query = mx.random.normal(shape=(1, 4, second_key.shape[2], 4)).astype(mx.float32)
    dense_key, dense_value = cache.gather_dense()
    dense_output = flash_attention(
        query,
        dense_key,
        dense_value,
        mask="causal",
    )
    paged_output = paged_attention(
        query,
        metadata.key_pages,
        metadata.value_pages,
        metadata.block_table,
        metadata.context_lens,
        metadata.page_size,
        mask=metadata.mask,
    )

    assert metadata.block_table.tolist() == [[0, 1]]
    assert metadata.context_lens.tolist() == [6]
    assert metadata.key_pages.shape == (2, 2, page_size, 4)
    assert metadata.value_pages.shape == (2, 2, page_size, 4)
    assert_allclose(paged_output, dense_output, precision=mx.float32)


def test_task_2_batched_paged_attention_matches_dense_attention():
    page_size = 4
    pool = TinyKvPagedPool(page_size=page_size)
    first = TinyKvPagedCache(pool=pool)
    second = TinyKvPagedCache(pool=pool)
    first.update_and_fetch(*_random_chunk(3))
    second.update_and_fetch(*_random_chunk(6))

    batch = BatchingKvCache(max_active_requests=3, max_seq_len=16)
    batch.add_request(first, 0)
    batch.add_request(second, 2)

    keys = mx.zeros((3, 2, 1, 4), dtype=mx.float32)
    values = mx.zeros((3, 2, 1, 4), dtype=mx.float32)
    keys[0:1], values[0:1] = _random_chunk(1)
    keys[2:3], values[2:3] = _random_chunk(1)

    metadata = batch.update_and_fetch_paged(
        keys,
        values,
        mask_length=1,
        mask="causal",
    )
    query = mx.random.normal(shape=(3, 4, 1, 4)).astype(mx.float32)
    paged_output = paged_attention(
        query,
        metadata.key_pages,
        metadata.value_pages,
        metadata.block_table,
        metadata.context_lens,
        metadata.page_size,
        mask=metadata.mask,
    )

    first_key, first_value = first.gather_dense()
    first_output = scaled_dot_product_attention_grouped(
        query[0:1], first_key, first_value, mask="causal"
    )
    second_key, second_value = second.gather_dense()
    second_output = scaled_dot_product_attention_grouped(
        query[2:3], second_key, second_value, mask="causal"
    )

    assert metadata.context_lens.tolist() == [4, 0, 7]
    assert metadata.block_table.shape == (3, 2)
    assert metadata.block_table.tolist()[1] == [-1, -1]
    assert metadata.key_pages.shape == (3, 2, page_size, 4)
    assert_allclose(paged_output[0:1], first_output, precision=mx.float32)
    assert_allclose(paged_output[2:3], second_output, precision=mx.float32)


def test_task_3_incremental_decode_matches_week2_with_paged_attention():
    mlx_model = _fake_qwen3_mlx_model()
    week2_model = Qwen3ModelWeek2(mlx_model)
    week3_model = Qwen3ModelWeek3(mlx_model, page_size=4)
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
            week3_out, week2_out, precision=mx.bfloat16, rtol=1e-3, atol=1e-3
        )
