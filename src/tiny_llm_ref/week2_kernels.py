import mlx.core as mx


class FastRMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


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
        if isinstance(offset, list):
            offset = mx.array(offset, dtype=mx.int32)
        x = x.transpose(0, 2, 1, 3)
        x = mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=self.base,
            scale=1.0,
            offset=offset,
        )
        return x.transpose(0, 2, 1, 3)


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    return gate * mx.sigmoid(gate) * up


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    return mx.fast.scaled_dot_product_attention(
        q=query,
        k=key,
        v=value,
        scale=scale,
        mask=mask,
    )
