import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        pass

    def __call__(
        self, x: mx.array, offset: slice | None = None
    ) -> mx.array:
        pass
