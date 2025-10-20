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
        self.seq_len = seq_len
        self.traditional = traditional

        # column vector of [0, ..., seq_len]
        p_vec = mx.arange(seq_len).reshape(-1, 1)
        # theta freq component that varies with embedding vec position i
        theta_vec = mx.reciprocal(
            mx.power(base, 2 * mx.arange(0, dims // 2) / dims)
        ).reshape(1, -1)
        # freqs shape is (seq_len, 1, dims / 2)
        freqs = mx.matmul(p_vec, theta_vec).reshape(seq_len, 1, -1)

        # shape of each of cos and sin freqs are (seq_len, 1, dims / 2)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        # x is shape (N, L, H, D).
        nlh_dims = x.shape[:-1]
        seq_len = min(nlh_dims[1], self.seq_len)
        # (N, L, H, D) -> (N, L, H, D//2, 2)
        x = x.reshape(*nlh_dims, self.dims // 2, 2)

        if offset:
            cos_freqs = self.cos_freqs[offset, :]
            sin_freqs = self.sin_freqs[offset, :]
        else:
            cos_freqs = self.cos_freqs[:seq_len, :]
            sin_freqs = self.sin_freqs[:seq_len, :]

        x_left, x_right = x[..., 0], x[..., 1]
        # shape: (N, L, H, D//2)
        out_left = mx.multiply(x_left, cos_freqs) - mx.multiply(x_right, sin_freqs)
        # shape: (N, L, H, D//2)
        out_right = mx.multiply(x_left, sin_freqs) + mx.multiply(x_right, cos_freqs)
        # shape: (N, L, H, D//2, 2)
        out = mx.stack([out_left, out_right], axis=-1)
        # shape: (N, L, H, D)
        out = out.reshape(*nlh_dims, -1)

        return out
