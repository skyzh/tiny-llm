import mlx.core as mx

from .quantize import QuantizedWeights


class FastRMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        pass

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


def supports_fused_gate_up(
    x: mx.array,
    w_gate: mx.array | QuantizedWeights,
    w_up: mx.array | QuantizedWeights,
) -> bool:
    pass


def quantized_gate_up_swiglu(
    x: mx.array,
    w_gate: mx.array | QuantizedWeights,
    w_up: mx.array | QuantizedWeights,
) -> mx.array:
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
    pass


def long_context_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: str | None = None,
) -> mx.array:
    pass
