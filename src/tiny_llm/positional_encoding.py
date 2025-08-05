import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        # refer to the implmentation of pytorch:
        # https://docs.pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
        self.D = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        # mx.arange(0, dims, 2) -> [0, 2, 4, 6, ..., self.dim - 2]
        theta = 1.0 / (
            base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        )
        # seq_idx = [0, 1, ..., seq_len - 1]
        seq_idx = mx.arange(seq_len, dtype=mx.float32)  # (L,)

        # Outer product: idx_theta[i, j] = seq_idx[i] * theta[j]
        idx_theta = seq_idx[:, None] * theta[None, :]  # (L, D/2)

        # Precompute cos/sin matrices
        self.cos_cached = mx.cos(idx_theta)  # (L, D/2)
        self.sin_cached = mx.sin(idx_theta)  # (L, D/2)\
        # TODO: implement the faster version for sparse matrix

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        """
        Args:
            x: input tensor of shape (N, L, H, D)
            offset: optional position slice(s), used during inference for sliding windows
        Returns:
            mx.array: output tensor with same shape as x
        """

        # x: (N, L, H, D)
        N, L, H, D = x.shape
        assert D == self.D, f"Expected dim={self.D}, got {D}"
        assert D % 2 == 0, "D must be even for RoPE"

        # Slice cached frequencies if offset is given
        cos = self.cos_cached[offset if offset else slice(0, L)]  # (L, D/2)
        sin = self.sin_cached[offset if offset else slice(0, L)]  # (L, D/2)

        # Expand cos/sin for broadcasting to (N, L, H, D/2)
        cos = cos[None, :, None, :]  # (1, L, 1, D/2)
        sin = sin[None, :, None, :]  # (1, L, 1, D/2)

        # Split x into even/odd channels
        x = x.astype(mx.float32)
        x1, x2 = x[..., ::2], x[..., 1::2]  # both (N, L, H, D/2)

        # Apply rotary transformation
        rotated = mx.stack([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], axis=-1)  # (N, L, H, D/2, 2)

        return rotated.reshape(N, L, H, D)