import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x_float = x.astype(mx.float32)
        means_squared = mx.mean(mx.square(x_float), axis=-1, keepdims=True)
        x_normalized = x_float * mx.rsqrt(means_squared + self.eps)
        return x_normalized.astype(input_dtype) * self.weight.astype(input_dtype)
