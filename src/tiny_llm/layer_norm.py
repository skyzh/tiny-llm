import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        
        x = x * mx.rsqrt(mx.mean(x.square(), axis=-1, keepdims=True) + self.eps)
        x = x * self.weight.astype(mx.float32)
        return x
