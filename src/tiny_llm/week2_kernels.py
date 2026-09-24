import mlx.core as mx
from extensions.tiny_llm_ext import _ext as tiny_llm_ext_private


_NO_ATTENTION_MASK = mx.zeros((1,), dtype=mx.float32)
DENSE_PREFILL_MIN_QUERY = 9


class FastRMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps
        self.dispatch_counts = {
            "register_cached": 0,
            "fixed_width_fallback": 0,
        }

    def __call__(self, x: mx.array) -> mx.array:
        pass


class FastRoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        pass

    def __call__(self, x: mx.array, offset: int | list[int] | mx.array = 0) -> mx.array:
        pass


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    pass


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def decode_attention_custom(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    # The bounded decode experiment is not learner-owned. Keep this compatibility
    # name on the readable path for Week 3 callers.
    return scaled_dot_product_attention(query, key, value, scale, mask)


def _prepare_dense_attention_inputs(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: mx.array | str | None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, bool, bool]:
    """Validate and flatten grouped-query inputs for the Day 5 native seam."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("dense attention expects [B,H,L,D] query, key, and value")
    batch_size, num_heads, query_length, head_dim = query.shape
    key_batch_size, num_kv_heads, context_length, key_head_dim = key.shape
    if batch_size != key_batch_size or key.shape != value.shape:
        raise ValueError("query, key, and value batch dimensions must match")
    if head_dim != key_head_dim or num_heads % num_kv_heads != 0:
        raise ValueError("incompatible grouped-query attention shapes")
    if isinstance(mask, str) and mask != "causal":
        raise ValueError(f"unsupported attention mask: {mask}")

    query = mx.contiguous(query.reshape(batch_size * num_heads, query_length, head_dim))
    key = mx.contiguous(
        key.reshape(batch_size * num_kv_heads, context_length, head_dim)
    )
    value = mx.contiguous(
        value.reshape(batch_size * num_kv_heads, context_length, head_dim)
    )
    is_causal = isinstance(mask, str) and mask == "causal"
    has_mask = isinstance(mask, mx.array)
    if has_mask:
        mask = mx.broadcast_to(
            mask, (batch_size, num_heads, query_length, context_length)
        )
        mask = mx.contiguous(
            mask.astype(mx.float32).reshape(
                batch_size * num_heads, query_length, context_length
            )
        )
    else:
        mask = _NO_ATTENTION_MASK
    return query, key, value, mask, is_causal, has_mask


def dense_prefill_attention_mma(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Run the learner-owned BQ32/BK16 online-softmax prefill kernel."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("dense attention expects [B,H,L,D] query, key, and value")
    expected_shape = query.shape
    _, num_heads, query_length, head_dim = query.shape
    num_kv_heads = key.shape[1]
    if query.dtype != mx.bfloat16 or head_dim != 128:
        raise ValueError("tiled dense prefill requires BF16 query/key/value with D=128")
    if query_length < DENSE_PREFILL_MIN_QUERY:
        raise ValueError(f"tiled dense prefill requires L >= {DENSE_PREFILL_MIN_QUERY}")
    query, key, value, mask, is_causal, has_mask = _prepare_dense_attention_inputs(
        query, key, value, mask
    )
    result = tiny_llm_ext_private._dense_attention_prefill_mma(
        query,
        key,
        value,
        mask,
        scale,
        is_causal,
        has_mask,
        num_heads,
        num_kv_heads,
    )
    return result.reshape(expected_shape)
