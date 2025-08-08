from typing import Optional

import mlx.core as mx


class TinyKvCache:
    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        pass


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        pass

    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        pass

    def add_request(self, prefilled: TinyKvCache, id: int):
        pass

    def remove_request(self, id: int):
        pass


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        pass

    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        pass


class TinyKvRotatingCache(TinyKvCache):
    def __init__(self, max_seq_len: int):
        pass

    def update_and_fetch(
        self, key: mx.array, value: mx.array, offset: int
    ) -> tuple[mx.array, mx.array]:
        pass
