import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.traditional = traditional

        half_dims = dims // 2
        angular_rates = mx.power(
            base, -mx.arange(half_dims, dtype=mx.float32) / half_dims
        )
        positions = mx.arange(seq_len, dtype=mx.float32)
        angles = mx.outer(positions, angular_rates)

        self.cos_freqs = mx.cos(angles)
        self.sin_freqs = mx.sin(angles)


    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        if isinstance(offset, list):
            raise NotImplementedError("Week 1 supports a single offset slice")

        sequence_length = x.shape[1]
        half_dims = self.dims // 2
        if offset is None:
            offset = slice(0, sequence_length)

        basis_shape = (1, sequence_length, 1, half_dims)
        cos_freqs = self.cos_freqs[offset].reshape(basis_shape)
        sin_freqs = self.sin_freqs[offset].reshape(basis_shape)
        working = x.astype(mx.float32)

        if (self.traditional):
            pairs = working.reshape(*working.shape[:-1], half_dims, 2)
            first = pairs[..., 0]
            second = pairs[..., 1]
        else:
            first = working[..., 0:half_dims]
            second = working[..., half_dims:self.dims]

        rotated_first = first * cos_freqs - second * sin_freqs
        rotated_second = first * sin_freqs + second * cos_freqs

        if(self.traditional):
            output = mx.stack([rotated_first, rotated_second], axis=-1).reshape(x.shape)
        else:
            output = mx.concat([rotated_first, rotated_second], axis=-1)

        return output.astype(x.dtype)

        
