import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        return (
            self.weight
            * x
            * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        ).astype(orig_dtype)
