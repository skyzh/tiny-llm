import mlx.core as mx
from .basics import softmax, linear
from extensions_ref import tiny_llm_ext_ref


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """
    A simple implementation of scaled dot product attention. Assuming Q, K, V are of the same shape.
    Assuming mask is always a float array that you can add to the scores.
    """
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L, S)), k=(S - L))
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """
    Potential input of the mask:
    - mx.array that can broadcast to B * H_q * L * S, which needs to be reshaped to match multi-head dimensions
    - None which will be ignored
    """
    factor = mx.rsqrt(query.shape[-1]) if scale is None else mx.array(scale)
    factor = factor.astype(query.dtype)
    expected_shape = query.shape

    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    B = query.shape[:-3]
    assert H_q % H == 0
    n_repeats = H_q // H

    query = query.reshape(*B, -1, H, n_repeats, L, D)
    key = key.reshape(*B, -1, H, 1, S, D)
    value = value.reshape(*B, -1, H, 1, S, D)

    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, scores.dtype)
            scores = scores + mask
        else:
            mask = mx.broadcast_to(mask, (*B, H_q, L, S))
            mask = mask.reshape(*B, 1, H, n_repeats, L, S)
            scores = scores + mask
    result = mx.matmul(softmax(scores, axis=-1), value)
    return result.reshape(expected_shape)


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """
    Paged attention backed by the C++/Metal extension.

    The Python wrapper keeps the model-facing shape as [B, H_q, L, D], while
    the extension sees flattened query heads and contiguous page storage.
    """
    if isinstance(mask, mx.array):
        raise NotImplementedError("Paged attention only supports mask=None or causal")
    if mask is not None and mask != "causal":
        raise NotImplementedError

    if len(query.shape) != 4:
        raise ValueError("query must be 4D [B, H_q, L, D]")
    if len(key_pages.shape) != 4 or len(value_pages.shape) != 4:
        raise ValueError("page tensors must be 4D [P, H_kv, page_size, D]")
    if key_pages.shape != value_pages.shape:
        raise ValueError("key pages and value pages must have the same shape")
    if len(block_table.shape) != 2 or len(context_lens.shape) != 1:
        raise ValueError("block_table must be 2D and context_lens must be 1D")
    if block_table.dtype != mx.int32 or context_lens.dtype != mx.int32:
        raise ValueError("block_table and context_lens must be int32")
    if not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("page_size must be a positive integer")

    factor = query.shape[-1] ** -0.5 if scale is None else float(scale)
    B, H_q, L, D = query.shape
    num_physical_pages, H, stored_page_size, stored_dim = key_pages.shape
    if min(B, H_q, L, D, H, stored_page_size, stored_dim) <= 0:
        raise ValueError("paged attention dimensions must be positive")
    if num_physical_pages <= 0:
        raise ValueError("paged attention requires nonempty physical page storage")
    if H_q % H != 0:
        raise ValueError("query heads must be divisible by K/V heads")
    if stored_dim != D:
        raise ValueError("query and page tensors must have the same head dimension")
    if stored_page_size != page_size:
        raise ValueError(
            f"page_size={page_size} does not match page storage {stored_page_size}"
        )
    if block_table.shape[0] != B or context_lens.shape[0] != B:
        raise ValueError("query, block_table, and context_lens batch sizes must match")
    max_pages = block_table.shape[1]
    if max_pages <= 0:
        raise ValueError("block_table must provide at least one page slot")

    if query.dtype != key_pages.dtype or query.dtype != value_pages.dtype:
        raise ValueError("query, key pages, and value pages must have the same dtype")
    if query.dtype not in (mx.float32, mx.bfloat16):
        raise ValueError("paged attention supports float32 or bfloat16 inputs")

    # Materialize and validate the small metadata tensors before the extension
    # can dispatch either its direct-decode or FlashAttention Metal branch.
    context_values = context_lens.tolist()
    block_rows = block_table.tolist()
    live_page_ids = set()
    for batch_idx, (context_len, row) in enumerate(zip(context_values, block_rows)):
        if context_len < 0:
            raise ValueError(f"context_lens[{batch_idx}] must be nonnegative")
        live_pages = (context_len + page_size - 1) // page_size
        if live_pages > max_pages:
            raise ValueError(f"context_lens[{batch_idx}] is not covered by block_table")
        for logical_page, page_id in enumerate(row):
            if logical_page < live_pages:
                if page_id < 0 or page_id >= num_physical_pages:
                    raise ValueError(
                        f"Live page id {page_id} at [{batch_idx}, {logical_page}] "
                        "is outside physical page storage"
                    )
                if page_id in live_page_ids:
                    raise ValueError(f"Live page id {page_id} is aliased")
                live_page_ids.add(page_id)
            elif page_id != -1:
                raise ValueError(
                    f"Unused block_table entry [{batch_idx}, {logical_page}] "
                    "must use the -1 sentinel"
                )
        if 0 < context_len < L:
            raise ValueError(
                f"context_lens[{batch_idx}] must be zero or at least query length {L}"
            )

    query = mx.contiguous(query.reshape(B * H_q, L, D))
    key_pages = mx.contiguous(key_pages)
    value_pages = mx.contiguous(value_pages)
    block_table = mx.contiguous(block_table)
    context_lens = mx.contiguous(context_lens)
    is_causal = mask == "causal"

    result = tiny_llm_ext_ref.paged_attention(
        query,
        key_pages,
        value_pages,
        block_table,
        context_lens,
        factor,
        is_causal=is_causal,
        num_kv_heads=H,
        num_heads=H_q,
    )
    return mx.contiguous(result.reshape(B, H_q, L, D))


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        assert wq.shape == (num_heads * self.head_dim, hidden_size)
        assert wk.shape == (num_heads * self.head_dim, hidden_size)
        assert wv.shape == (num_heads * self.head_dim, hidden_size)
        assert wo.shape == (hidden_size, num_heads * self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        assert query.shape == key.shape == value.shape
        projection_q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        x = scaled_dot_product_attention_simple(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        )
        x = x.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_size)
        return linear(x, self.wo)
