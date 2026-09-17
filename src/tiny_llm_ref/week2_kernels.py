import mlx.core as mx
from extensions_ref import tiny_llm_ext_ref

from .basics import softmax
from .quantize import QuantizedWeights


_NO_ATTENTION_MASK = mx.zeros((1,), dtype=mx.float32)


class FastRMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return tiny_llm_ext_ref.rms_norm(
            mx.contiguous(x), mx.contiguous(self.weight.astype(x.dtype)), self.eps
        )


class FastRoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

    def __call__(self, x: mx.array, offset: int | list[int] | mx.array = 0) -> mx.array:
        batch_size = x.shape[0]
        if isinstance(offset, int):
            offset = mx.full((batch_size,), offset, dtype=mx.int32)
        elif isinstance(offset, list):
            if len(offset) != batch_size:
                raise ValueError("FastRoPE needs one offset per batch row")
            offset = mx.array(offset, dtype=mx.int32)
        elif offset.ndim == 0:
            offset = mx.broadcast_to(offset.astype(mx.int32), (batch_size,))
        elif offset.shape != (batch_size,):
            raise ValueError("FastRoPE needs one offset per batch row")
        return tiny_llm_ext_ref.rope(
            mx.contiguous(x),
            mx.contiguous(offset.astype(mx.int32)),
            self.dims,
            self.base,
            self.traditional,
        )


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    return tiny_llm_ext_ref.swiglu(mx.contiguous(gate), mx.contiguous(up))


def supports_fused_gate_up(
    x: mx.array,
    w_gate: mx.array | QuantizedWeights,
    w_up: mx.array | QuantizedWeights,
) -> bool:
    if x.ndim < 2:
        return False
    if not isinstance(w_gate, QuantizedWeights) or not isinstance(
        w_up, QuantizedWeights
    ):
        return False
    rows = 1
    for size in x.shape[:-1]:
        rows *= size
    input_dim = x.shape[-1]
    return (
        x.dtype == mx.bfloat16
        and 0 < rows <= 2048
        and input_dim % 128 == 0
        and w_gate.group_size == 128
        and w_up.group_size == 128
        and w_gate.bits == 4
        and w_up.bits == 4
        and w_gate.biases is not None
        and w_up.biases is not None
        and w_gate.weight.dtype == mx.uint32
        and w_up.weight.dtype == mx.uint32
        and w_gate.scales.dtype == mx.bfloat16
        and w_up.scales.dtype == mx.bfloat16
        and w_gate.biases.dtype == mx.bfloat16
        and w_up.biases.dtype == mx.bfloat16
        and w_gate.weight.shape == w_up.weight.shape
        and w_gate.weight.shape[0] > 0
        and w_gate.scales.shape == w_up.scales.shape
        and w_gate.biases.shape == w_gate.scales.shape
        and w_up.biases.shape == w_up.scales.shape
        and w_gate.weight.shape[1] * 8 == input_dim
        and w_gate.scales.shape == (w_gate.weight.shape[0], input_dim // 128)
    )


def quantized_gate_up_swiglu(
    x: mx.array,
    w_gate: mx.array | QuantizedWeights,
    w_up: mx.array | QuantizedWeights,
) -> mx.array:
    if not supports_fused_gate_up(x, w_gate, w_up):
        raise ValueError(
            "quantized_gate_up_swiglu requires BF16 input and matching packed "
            "4-bit group-128 gate/up weights with 1..2048 rows"
        )
    leading_shape = x.shape[:-1]
    input_dim = x.shape[-1]
    output = tiny_llm_ext_ref.quantized_gate_up_swiglu(
        mx.contiguous(x.reshape(-1, input_dim)),
        mx.contiguous(w_gate.scales),
        mx.contiguous(w_gate.biases),
        mx.contiguous(w_gate.weight),
        mx.contiguous(w_up.scales),
        mx.contiguous(w_up.biases),
        mx.contiguous(w_up.weight),
        128,
        4,
    )
    return output.reshape(*leading_shape, w_gate.weight.shape[0])


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    expected_shape = query.shape
    batch_shape = query.shape[:-3]
    num_heads, query_length, head_dim = query.shape[-3:]
    num_kv_heads, context_length, _ = key.shape[-3:]
    if key.shape != value.shape or num_heads % num_kv_heads != 0:
        raise ValueError("incompatible grouped-query attention shapes")

    repeats = num_heads // num_kv_heads
    query = query.reshape(
        *batch_shape, -1, num_kv_heads, repeats, query_length, head_dim
    )
    key = key.reshape(*batch_shape, -1, num_kv_heads, 1, context_length, head_dim)
    value = value.reshape(*batch_shape, -1, num_kv_heads, 1, context_length, head_dim)
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * mx.array(scale, dtype=query.dtype)
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"unsupported attention mask: {mask}")
        causal = mx.tril(
            mx.ones((query_length, context_length)), k=context_length - query_length
        )
        scores = scores + mx.where(causal, 0, -mx.inf).astype(scores.dtype)
    elif mask is not None:
        mask = mx.broadcast_to(
            mask, (*batch_shape, num_heads, query_length, context_length)
        )
        scores = scores + mask.reshape(
            *batch_shape, -1, num_kv_heads, repeats, query_length, context_length
        )
    return mx.matmul(softmax(scores, axis=-1), value).reshape(expected_shape)


def decode_attention_custom(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
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

    result = tiny_llm_ext_ref.decode_attention(
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
    return result.reshape(batch_size, num_heads, query_length, head_dim)


def long_context_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: str | None = None,
) -> mx.array:
    if isinstance(mask, str) and mask != "causal":
        raise ValueError(f"unsupported attention mask: {mask}")
    if isinstance(mask, mx.array):
        raise ValueError("long-context attention does not accept explicit masks")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "long-context attention requires BF16 Qwen3-4B dense GQA "
            "Q=[B,32,1|2,128], KV=[B,8,S<=32768,128]"
        )
    batch_size, num_heads, query_length, head_dim = query.shape
    key_batch_size, num_kv_heads, context_length, key_head_dim = key.shape
    if (
        query.dtype != mx.bfloat16
        or key.dtype != mx.bfloat16
        or value.dtype != mx.bfloat16
        or batch_size != key_batch_size
        or key.shape != value.shape
        or num_heads != 32
        or num_kv_heads != 8
        or head_dim != 128
        or key_head_dim != 128
        or query_length not in (1, 2)
        or not 0 < context_length <= 32768
    ):
        raise ValueError(
            "long-context attention requires BF16 Qwen3-4B dense GQA "
            "Q=[B,32,1|2,128], KV=[B,8,S<=32768,128]"
        )
    flat_query = mx.contiguous(
        query.reshape(batch_size * num_heads, query_length, head_dim)
    )
    flat_key = mx.contiguous(
        key.reshape(batch_size * num_kv_heads, context_length, head_dim)
    )
    flat_value = mx.contiguous(
        value.reshape(batch_size * num_kv_heads, context_length, head_dim)
    )
    result = tiny_llm_ext_ref.long_context_attention(
        flat_query,
        flat_key,
        flat_value,
        _NO_ATTENTION_MASK,
        scale,
        isinstance(mask, str),
        num_heads,
        num_kv_heads,
    )
    return result.reshape(batch_size, num_heads, query_length, head_dim)
