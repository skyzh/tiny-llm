import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        half_dims = dims // 2
        self.half_dims = half_dims
        self.traditional = traditional
        pos = mx.arange(seq_len)    # pos = [0, 1, 2, ..., L-1]
        inner = mx.arange(0, half_dims, dtype=mx.float32) / half_dims   # [2i/d], i=0,...,half_dims-1
        w =  mx.power(base, -inner) # [1/10000^(i/half_dims)], i=0,...,half_dims-1
        theta = mx.outer(pos, w)    # (seq_len, half_dims)
        self.cos_theta = mx.cos(theta)
        self.sin_theta = mx.sin(theta)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, S, H, D = x.shape

        # apply offset
        if offset is not None:
            if isinstance(offset, slice):
                assert offset.stop - offset.start == S, f"offset must be of length {S}"
            elif isinstance(offset, list):
                assert len(offset) == N, (
                    f"offsets must have the same length as batch size {N}"
                )
                for o in offset:
                    assert o.stop - o.start == S, f"offset must be of length {S}"
                offset = mx.array([list(range(i.start, i.stop)) for i in offset])
        cos_biasis = (self.cos_theta[:S, :] if offset is None else self.cos_theta[offset, :])
        sin_biasis = (self.sin_theta[:S, :] if offset is None else self.sin_theta[offset, :])

        # reshape x: (N, S, H, D // 2, 2)
        if self.traditional:    # [0, 2, 4, 6] [1, 3, 5, 7] format
            x = x.reshape(N, S, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:                   # [0, 1, 2, 3] [4, 5, 6, 7] format
            x1 = x[..., 0 : self.half_dims]
            x2 = x[..., self.half_dims : D]
        
        # reshape basis: (N, S, 1, half_dims)
        cos_biasis = cos_biasis.reshape(-1, S, 1, self.half_dims)
        sin_biasis = sin_biasis.reshape(-1, S, 1, self.half_dims)

        # [real; imag] = [cos -sin; sin cos] * [x1; x2]
        real = mx.multiply(x1, cos_biasis) - mx.multiply(x2, sin_biasis)
        imag = mx.multiply(x2, cos_biasis) + mx.multiply(x1, sin_biasis)
        if self.traditional:
            y = mx.stack([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)
        else:
            y = mx.concat([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)
        return y.astype(x.dtype)
