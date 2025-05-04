import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.seq_len = seq_len
        half_dims = dims // 2
        inner = mx.arange(0, half_dims, dtype=mx.float32) / half_dims
        freqs = mx.power(base, -inner)
        t = mx.arange(seq_len)
        freqs = mx.outer(t, freqs)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)
        self.base = base
        self.half_dims = half_dims
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: slice | None = None
    ) -> mx.array:
        N, S, H, D = x.shape
        # if offset is not None:
        #     assert len(offset) == S, f"offset {len(offset)} must be of length {s}"
        cos_basis = (
            self.cos_freqs[:S, :] if offset is None else self.cos_freqs[offset, :]
        )
        sin_basis = (
            self.sin_freqs[:S, :] if offset is None else self.sin_freqs[offset, :]
        )
        # reshape x: (b, s, n_heads, head_dim // 2, 2)
        if self.traditional:
            x = x.reshape(N, S, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            x1 = x[..., 0 : self.half_dims]
            x2 = x[..., self.half_dims : self.dims]
        # reshape basis: (1, s, 1, dims // 2, 2)
        cos_basis = cos_basis.reshape(S, 1, self.half_dims)
        sin_basis = sin_basis.reshape(S, 1, self.half_dims)
        # manually doing complex number multiplication..
        real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
        imag = mx.multiply(x2, cos_basis) + mx.multiply(x1, sin_basis)
        if self.traditional:
            y = mx.stack([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)
        else:
            y = mx.concat([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)
        return y
