import mlx.core as mx

class RoPE:
    _RoPE_cache_key = None
    _RoPE_cache_value = None

    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.max_seq_len = seq_len
        self.base = base
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape
        assert D == self.dims

        costh, sinth = RoPE._cal_cos_sin_theta(offset, L, self.base, self.dims)
        costh = costh[:, None, :]
        sinth = sinth[:, None, :]

        if self.traditional:
            x = x.reshape(-1, H, D)
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            rx1 = costh * x1 - sinth * x2
            rx2 = sinth * x1 + costh * x2
            rx = mx.concatenate([rx1[..., None], rx2[..., None]], axis=-1).reshape(N, L, H, D)

        else:
            x = x.reshape(-1, H, D)
            x1 = x[..., : D//2]
            x2 = x[..., D//2:]
            rx1 = costh * x1 - sinth * x2
            rx2 = sinth * x1 + costh * x2
            rx = mx.concatenate([rx1, rx2], axis=-1).reshape(N, L, H, D)
        return rx

    @classmethod
    def _cal_cos_sin_theta(cls, offset, L, base, dim): 
        if (offset, L, base, dim) == cls._RoPE_cache_key:
            return cls._RoPE_cache_value
        
        if offset is None:
            off = list(range(0, L))
        elif type(offset) is slice:
            off = list(range(offset.start or 0, offset.stop, offset.step or 1))
            assert len(off) == L
        
        pos = mx.array(off, dtype=mx.float32)

        d = dim // 2
        
        freq = mx.exp(
            -mx.arange(0.0, d) * (mx.log(base) / d)
        )

        theta = pos.reshape(-1, 1) * freq.reshape(1, -1)
        cos_theta = mx.cos(theta)
        sin_theta = mx.sin(theta)
        
        assert(cos_theta.shape == (L, d))
        assert(sin_theta.shape == (L, d))

        cls._RoPE_cache_key = (offset, L, base, dim)
        cls._RoPE_cache_value = (cos_theta, sin_theta)

        return cos_theta, sin_theta
