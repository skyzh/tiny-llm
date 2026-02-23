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
        freqs = mx.power(base, mx.arange(0, dims // 2) * -2 / dims)
        self.theta = mx.outer(mx.arange(seq_len), freqs)
        self.cos = mx.cos(self.theta)
        self.sin = mx.sin(self.theta)

        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, S, H, D = x.shape
        half_dims = self.dims // 2
        if self.traditional:
            even = x[..., 0::2]
            odd = x[..., 1::2]
        else:
            even = x[..., :half_dims]
            odd = x[..., half_dims:]

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
            else:
                raise ValueError("offset must be a slice or a list of slices")

        cos = self.cos[:S, :] if offset is None else self.cos[offset, :]
        sin = self.sin[:S, :] if offset is None else self.sin[offset, :]

        cos = cos.reshape(-1, S, 1, half_dims)
        sin = sin.reshape(-1, S, 1, half_dims)
        r = [even * cos - odd * sin, odd * cos + even * sin]
        if self.traditional:
            r = mx.stack(r, axis=-1)
        else:
            r = mx.concat(r, axis=-1)
        return r.reshape(N, S, H, D)  # zip even and odd
