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
        freqs = mx.outer(mx.arange(seq_len), mx.power(base, -mx.arange(0, dims//2, dtype=mx.float32) / (dims//2)))
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)
        self.base = base
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, S, H, D = x.shape
        if offset is not None:
            if isinstance(offset, slice):
                assert offset.stop - offset.start == S, f"offset must be of length {S}"
            elif isinstance(offset, list):
                assert len(offset) == N
                for o in offset:
                    assert o.stop - o.start == S, f"offset must be of length {S}"
                offset = mx.array([list(range(i.start, i.stop)) for i in offset])

        cos_basic = (self.cos_freqs[:S, :] if offset is None else self.cos_freqs[offset, :])
        sin_basic = (self.sin_freqs[:S, :] if offset is None else self.sin_freqs[offset, :])
        
        if self.traditional:
            x = x.reshape(N, S, H, self.dims//2, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            x1 = x[..., 0 : self.dims // 2]
            x2 = x[..., self.dims // 2 : self.dims]
        
        cos_basic = cos_basic.reshape(-1, S, 1, self.dims // 2)
        sin_basic = sin_basic.reshape(-1, S, 1, self.dims // 2)
        real = mx.multiply(x1, cos_basic) - mx.multiply(x2, sin_basic)
        imag = mx.multiply(x2, cos_basic) + mx.multiply(x1, sin_basic)
        if self.traditional:
            return mx.stack([real, imag], axis=-1).reshape(N, S, H, D).astype(x.dtype)
        else:
            return mx.concat([real, imag], axis=-1).reshape(N, S, H, D).astype(x.dtype)
        