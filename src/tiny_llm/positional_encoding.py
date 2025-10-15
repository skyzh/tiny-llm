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
        # freqs shape is (seq_len, dims / 2)
        # freqs = mx.matmul(p_vec, theta_vec)

        # shape of each of cos and sin freqs are (seq_len, 1, dims / 2)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        # x is shape (N, L, H, D). we want it to be (N, H, L, D // 2, 2)
        nlh_dims = x.shape[:-1]
        seq_len = min(nlh_dims[1], self.seq_len)
        # (N, L, H, D) -> (N, L, H, D//2, 2)
        x = x.reshape(*nlh_dims, self.dims // 2, 2)
        # (N, L, H, D//2, 2) -> (N, H, L, D//2, 2)
        # x = x.swapaxes(-3, -4)

        # shape: (N, H, L, D//2)
        out_left = mx.multiply(x[..., 0], self.cos_freqs[:seq_len, :]) - mx.multiply(
            x[..., 1], self.sin_freqs[:seq_len, :]
        )
        # shape: (N, H, L, D//2)
        out_right = mx.multiply(x[..., 0], self.sin_freqs[:seq_len, :]) + mx.multiply(
            x[..., 1], self.cos_freqs[:seq_len, :]
        )
        # shape: (N, H, L, D//2, 2) -> (N, L, H, D//2, 2)
        # out = mx.stack([out_left, out_right], axis=-1).swapaxes(-3, -4)
        # shape: (N, L, H, D//2, 2)
        out = mx.stack([out_left, out_right], axis=-1)
        # shape: (N, L, H, D)
        out = out.reshape(*nlh_dims, -1)

        return out
