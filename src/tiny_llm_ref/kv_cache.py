from typing import Optional

from .attention import causal_mask
import mlx.core as mx


class TinyKvCache:
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of the updated key-value cache, the updated value, the sequence length, and the mask.
        """
        pass


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.kv_caches: list[TinyKvCache] = [None] * max_active_requests
        self.HD = None

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        B, H, S, D = keys.shape
        assert keys.shape == values.shape
        assert S <= self.max_seq_len
        assert self.HD == (H, D), f"expect {self.HD} but got {H, D}"
        assert B == self.max_active_requests
        # Step 1: append the result to the cache
        data = []
        for b in range(B):
            if self.kv_caches[b] is None:
                data.append(None)
                continue
            key, value = keys[b : b + 1], values[b : b + 1]
            new_key, new_value, seq_len, mask = self.kv_caches[b].update_and_fetch(
                key, value
            )
            data.append((new_key[0], new_value[0], seq_len, mask))

        # Step 2: compute seq_len of this batch
        def get_seq_len(data):
            if data is None:
                return 0
            _, _, seq_len, _ = data
            return seq_len

        seq_len = max(map(get_seq_len, data))
        # Step 3: generate masks and a single array of keys and values
        keys = mx.zeros((self.max_active_requests, H, seq_len, D), dtype=key.dtype)
        values = mx.zeros((self.max_active_requests, H, seq_len, D), dtype=value.dtype)
        masks = mx.full(
            (self.max_active_requests, mask_length, seq_len), -mx.inf, dtype=key.dtype
        )
        for b in range(B):
            if data[b] is None:
                # for some reasons we need to do this, otherwise it will cause wrong output?
                # maybe precision issues?
                masks[b, :, :] = causal_mask(mask_length, seq_len, dtype=key.dtype)
                continue
            key, value, S, mask = data[b]
            keys[b, :, seq_len - S : seq_len, :] = key
            values[b, :, seq_len - S : seq_len, :] = value
            if mask is None or mask == "causal":
                masks[b, :, seq_len - S : seq_len] = causal_mask(
                    mask_length, S, dtype=key.dtype
                )
            elif isinstance(mask, mx.array):
                masks[b, :, seq_len - S : seq_len] = mask
            else:
                raise NotImplemented
        return keys, values, None, masks.reshape(B, 1, mask_length, seq_len)

    def add_request(self, prefilled: TinyKvCache, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        keys, _ = prefilled.key_values
        B, H, _, D = keys.shape
        assert B == 1
        if self.HD is None:
            self.HD = (H, D)
        else:
            assert self.HD == (H, D)
        self.kv_caches[id] = prefilled

    def remove_request(self, id: int):
        if self.kv_caches is None:
            raise ValueError(f"Request id {id} is not in the cache")
        self.kv_caches[id] = None


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if self.key_values is None:
            assert self.offset == 0
            self.key_values = (key, value)
            B, H, S, D = key.shape
            self.offset = S
            return key, value, 0, mask
        else:
            B, H, S, D = key.shape
            assert key.shape == value.shape
            prev_keys, prev_values = self.key_values
            assert prev_keys.shape == (B, H, self.offset, D)
            assert prev_values.shape == (B, H, self.offset, D)
            new_keys = mx.concat([prev_keys, key], axis=2)
            new_values = mx.concat([prev_values, value], axis=2)
            self.key_values = (new_keys, new_values)
            self.offset += S
            return new_keys, new_values, self.offset, mask
