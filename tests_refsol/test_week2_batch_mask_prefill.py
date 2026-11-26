import mlx.core as mx
import pytest

from extensions_ref.tiny_llm_ext_ref import _ext
from tiny_llm_ref.week2_kernels import dense_prefill_attention_mma


def fixture(length, context, dtype, strided=False):
    def operand(heads, tokens, seed):
        return mx.random.normal(
            (2, heads, tokens, 128), dtype=mx.bfloat16, key=mx.random.key(seed)
        )

    query = operand(32, length, 1)
    key = operand(8, context, 2)
    value = operand(8, context, 3)
    query_position = mx.arange(length)[None, None, :, None]
    key_position = mx.arange(context)[None, None, None, :]
    batch = mx.arange(2)[:, None, None, None]
    valid = (key_position < context - batch * 4) & (query_position != length - 1)
    penalty = ((key_position + batch) % 7).astype(dtype) * -0.125
    mask = mx.where(valid, penalty, -mx.inf).astype(dtype)
    if strided:
        mask = mx.repeat(mask, 2, axis=-1)[..., ::2]
        value = value[:, :, ::-1, :]
    return query, key, value, mask


def bits_equal(left, right):
    return mx.array_equal(left.view(mx.uint16), right.view(mx.uint16)).item()


@pytest.mark.parametrize(
    "length,context,dtype,strided",
    [(17, 17, mx.float32, True), (193, 257, mx.bfloat16, False)],
)
def test_explicit_batch_mask_uses_one_plane_per_row(
    monkeypatch, length, context, dtype, strided
):
    query, key, value, mask = fixture(length, context, dtype, strided)
    stock_binding = _ext._dense_attention_prefill_mma
    batch_binding = _ext._dense_attention_prefill_mma_batch_mask
    calls = []

    def compact(q, k, v, mask_arg, *rest):
        calls.append(mask_arg.shape)
        assert mask_arg.dtype == mx.float32
        return batch_binding(q, k, v, mask_arg, *rest)

    def stock(q, k, v, mask_arg, *rest):
        calls.append(mask_arg.shape)
        assert mask_arg.dtype == mx.float32
        return stock_binding(q, k, v, mask_arg, *rest)

    monkeypatch.setattr(_ext, "_dense_attention_prefill_mma_batch_mask", compact)
    monkeypatch.setattr(_ext, "_dense_attention_prefill_mma", stock)
    actual = dense_prefill_attention_mma(query, key, value, 128**-0.5, mask)
    expanded = mx.broadcast_to(mask, (2, 32, length, context))
    expected = dense_prefill_attention_mma(query, key, value, 128**-0.5, expanded)
    assert calls == [(2, length, context), (64, length, context)]
    assert bits_equal(actual, expected)
    assert mx.all(actual[:, :, -1, :] == 0).item()
    assert not bits_equal(actual[0], actual[1])


def test_other_masks_and_missing_batch_binding_use_stock(monkeypatch):
    query, key, value, mask = fixture(17, 17, mx.float32)
    stock_binding = _ext._dense_attention_prefill_mma
    calls = []

    def stock(q, k, v, mask_arg, _scale, causal, has_mask, *rest):
        calls.append((mask_arg.shape, causal, has_mask))
        return stock_binding(q, k, v, mask_arg, _scale, causal, has_mask, *rest)

    def unexpected(*_args):
        pytest.fail("Unsupported layout used the batch mask binding")

    monkeypatch.setattr(_ext, "_dense_attention_prefill_mma_batch_mask", unexpected)
    monkeypatch.setattr(_ext, "_dense_attention_prefill_mma", stock)
    options = (
        mx.broadcast_to(mask, (2, 32, 17, 17)),
        mask[:1],
        mask.astype(mx.float16),
        "causal",
        None,
    )
    outputs = [
        dense_prefill_attention_mma(query, key, value, 128**-0.5, item)
        for item in options
    ]
    monkeypatch.setattr(_ext, "_dense_attention_prefill_mma_batch_mask", None)
    outputs.append(dense_prefill_attention_mma(query, key, value, 128**-0.5, mask))
    mx.eval(*outputs)
    assert calls[:3] == [((64, 17, 17), False, True)] * 3
    assert [(causal, has_mask) for _, causal, has_mask in calls[3:5]] == [
        (True, False),
        (False, False),
    ]
    assert calls[5:] == [((64, 17, 17), False, True)]
    assert bits_equal(outputs[0], outputs[-1])
