import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        # use high precision for mean and square
        x = x.astype(mx.float32)
        mean = mx.mean(mx.square(x), axis=-1, keepdims=True)
        return (x * self.weight / mx.sqrt(mean + self.eps)).astype(orig_dtype)
