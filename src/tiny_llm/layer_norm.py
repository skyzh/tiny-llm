import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        # weight: (D) which i guess is dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        floater = mx.array(1.0, dtype=mx.float32)
        # x: (N.., D)
        # scaling_factor: (N..)
        scaling_factor = mx.rsqrt(mx.mean(floater * x * x, axis=-1) + self.eps)
        # expand scaling factor to (N.., 1)
        # (N.., 1) * (N.., D) -> (N.., D)
        # (N.., D) * (D) -> (N.., D)
        return mx.expand_dims(scaling_factor, -1) * x * self.weight
