from abc import ABC, abstractmethod
from typing import Optional

from .attention import causal_mask
import mlx.core as mx


class TinyKvCache(ABC):
    @abstractmethod
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

    def release(self):
        """
        Release all resources owned by this cache.

        Request-scoped caches use this when generation finishes or a batch slot
        is removed. Dense caches do not own shared resources, while paged caches
        return their physical pages to a shared pool.
        """
        return None

    def update_and_fetch_paged(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        """
        Update this cache and return paged attention metadata.

        Week 3 caches override this. Dense caches intentionally do not provide
        paged metadata, so calling this on them is a programming error.
        """
        raise NotImplementedError("This KV cache does not support paged attention")

    def rewind(self, n: int):
        """
        Remove the newest n logical tokens from this cache.

        This is needed by speculative decoding when some draft tokens are
        rejected after their K/V has already been written. Implementations may
        drop dense suffixes or return whole pages to a page pool.
        """
        raise NotImplementedError("This KV cache does not support rewind")


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
        if self.HD is None:
            self.HD = (H, D)
        else:
            assert self.HD == (H, D), f"expect {self.HD} but got {H, D}"
        assert B == self.max_active_requests
        # Step 1: append each active row into its request cache. This method
        # preserves the legacy dense batch interface for Week 2/Day 1 callers.
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
        # Step 3: rebuild one dense batch tensor. True paged attention will
        # replace this with block_table/context_lens metadata.
        keys = mx.zeros((self.max_active_requests, H, seq_len, D), dtype=key.dtype)
        values = mx.zeros((self.max_active_requests, H, seq_len, D), dtype=value.dtype)
        masks = mx.full(
            (self.max_active_requests, mask_length, seq_len), -mx.inf, dtype=key.dtype
        )
        for b in range(B):
            if data[b] is None:
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
                raise NotImplementedError
        return keys, values, None, masks.reshape(B, 1, mask_length, seq_len)

    def update_and_fetch_paged(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        from .paged_kv_cache import PagedKvMetadata, TinyKvPagedCache

        B, H, S, D = keys.shape
        assert keys.shape == values.shape
        assert S <= self.max_seq_len
        if self.HD is None:
            self.HD = (H, D)
        else:
            assert self.HD == (H, D), f"expect {self.HD} but got {H, D}"
        assert B == self.max_active_requests

        pool = None
        context_lens = []
        max_pages = 0
        for b in range(B):
            cache = self.kv_caches[b]
            if cache is None:
                context_lens.append(0)
                continue
            if not isinstance(cache, TinyKvPagedCache):
                raise ValueError("BatchingKvCache contains a non-paged request cache")
            cache.update_and_fetch_paged(
                keys[b : b + 1],
                values[b : b + 1],
                mask_length=mask_length,
                mask=mask,
            )
            if pool is None:
                pool = cache.pool
            elif pool is not cache.pool:
                raise ValueError("Paged batch caches must share one page pool")
            context_lens.append(cache.offset)
            max_pages = max(max_pages, cache.num_pages)

        if pool is None:
            raise ValueError("Cannot build paged metadata without active requests")

        rows = []
        for cache in self.kv_caches:
            if cache is None:
                rows.append([-1] * max_pages)
            else:
                rows.append(cache.page_ids + [-1] * (max_pages - cache.num_pages))

        return PagedKvMetadata(
            key_pages=pool.key_pages,
            value_pages=pool.value_pages,
            block_table=mx.array(rows, dtype=mx.int32),
            context_lens=mx.array(context_lens, dtype=mx.int32),
            page_size=pool.page_size,
            mask=mask,
        )

    def add_request(self, prefilled: TinyKvCache, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        if isinstance(prefilled, TinyKvFullCache) and prefilled.key_values is not None:
            keys, _ = prefilled.key_values
            B, H, _, D = keys.shape
            assert B == 1
            if self.HD is None:
                self.HD = (H, D)
            else:
                assert self.HD == (H, D)
        self.kv_caches[id] = prefilled

    def remove_request(self, id: int):
        if self.kv_caches[id] is None:
            raise ValueError(f"Request id {id} is not in the cache")
        self.kv_caches[id].release()
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
            return key, value, self.offset, mask
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

    def rewind(self, n: int):
        self.offset -= n
        self.key_values = (
            self.key_values[0][:, :, : self.offset],
            self.key_values[1][:, :, : self.offset],
        )
