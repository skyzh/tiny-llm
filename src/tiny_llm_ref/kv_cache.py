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

    def rewind(self, n: int):
        """
        Remove the newest n logical tokens from this cache.

        This is needed by speculative decoding when some draft tokens are
        rejected after their K/V has already been written. Implementations may
        drop dense suffixes or return whole pages to a page pool.
        """
        raise NotImplementedError("This KV cache does not support rewind")


class TinyKvPagedPool:
    """Model-local physical storage for paged KV.

    The model owns one pool and passes it to every layer cache. The pool gives
    out physical page ids from one free list. Because every live page id is
    unique, the page id alone is enough to find the physical K/V page.
    """

    def __init__(self, page_size: int = 128):
        assert page_size > 0
        self.page_size = page_size
        self.key_pages: list[mx.array | None] = []
        self.value_pages: list[mx.array | None] = []
        self.free_page_ids: list[int] = []
        self.used_page_ids: set[int] = set()

    @property
    def num_pages(self) -> int:
        return len(self.key_pages)

    @property
    def num_free_pages(self) -> int:
        return len(self.free_page_ids)

    def _check_page_chunk(self, x: mx.array) -> None:
        B, H, S, D = x.shape
        assert 0 < S <= self.page_size

    def allocate_page(self) -> int:
        # The page id is allocated from a model-wide free list. In this teaching
        # version, a layer cache owns the page until release/rewind returns it.
        if self.free_page_ids:
            page_id = self.free_page_ids.pop()
        else:
            page_id = self.num_pages
            self.key_pages.append(None)
            self.value_pages.append(None)
        self.used_page_ids.add(page_id)
        return page_id

    def read_page(self, page_id: int) -> tuple[mx.array, mx.array]:
        key = self.key_pages[page_id]
        value = self.value_pages[page_id]
        if key is None or value is None:
            raise ValueError(f"Page {page_id} has no storage")
        return key, value

    def _ensure_page_storage(
        self,
        page_id: int,
        key: mx.array,
        value: mx.array,
    ) -> tuple[mx.array, mx.array]:
        key_page = self.key_pages[page_id]
        value_page = self.value_pages[page_id]
        if key_page is not None and value_page is not None:
            return key_page, value_page

        B, H, _, D = key.shape
        key_page = mx.zeros((B, H, self.page_size, D), dtype=key.dtype)
        value_page = mx.zeros((B, H, self.page_size, D), dtype=value.dtype)
        self.key_pages[page_id] = key_page
        self.value_pages[page_id] = value_page
        return key_page, value_page

    def write_page_slice(
        self,
        page_id: int,
        start: int,
        key: mx.array,
        value: mx.array,
    ) -> None:
        assert key.shape == value.shape
        self._check_page_chunk(key)
        if page_id not in self.used_page_ids:
            raise ValueError(f"Page {page_id} is free")
        key_page, value_page = self._ensure_page_storage(page_id, key, value)
        B, H, capacity, D = key_page.shape
        assert value_page.shape == (B, H, capacity, D)
        assert capacity == self.page_size
        assert key.shape[:2] == (B, H)
        assert key.shape[3] == D
        end = start + key.shape[2]
        assert 0 <= start <= capacity
        assert end <= self.page_size

        key_page[:, :, start:end, :] = key
        value_page[:, :, start:end, :] = value
        self.key_pages[page_id] = key_page
        self.value_pages[page_id] = value_page

    def free_page(self, page_id: int) -> None:
        if page_id not in self.used_page_ids:
            raise ValueError(f"Page {page_id} is already free")
        # Keep the page id stable, but drop its old K/V tensors so the id can be
        # handed to a future cache.
        self.used_page_ids.remove(page_id)
        self.key_pages[page_id] = None
        self.value_pages[page_id] = None
        self.free_page_ids.append(page_id)


class TinyKvPagedCache(TinyKvCache):
    """Layer-local K/V cache backed by a model-owned page pool.

    Each transformer layer gets its own TinyKvPagedCache and therefore its own
    `page_ids`, `page_lens`, and `offset`. The shared part is only the pool,
    which lets pages be recycled across requests and layers.
    """

    def __init__(self, pool: TinyKvPagedPool):
        self.pool = pool
        self.page_size = self.pool.page_size
        self.page_ids: list[int] = []
        self.page_lens: list[int] = []
        self.offset = 0

    @property
    def num_pages(self) -> int:
        return len(self.page_ids)

    @property
    def key_values(self) -> tuple[mx.array, mx.array] | None:
        if self.offset == 0:
            return None
        return self.gather_dense()

    def _append_chunk(self, key: mx.array, value: mx.array) -> None:
        assert key.shape == value.shape
        B, H, S, D = key.shape
        assert B == 1, "Paged request cache only supports one request at a time"
        start = 0

        # First fill the existing tail page if it has free slots.
        if self.page_ids and self.page_lens[-1] < self.page_size:
            page_id = self.page_ids[-1]
            page_start = self.page_lens[-1]
            take = min(self.page_size - page_start, S)
            self.pool.write_page_slice(
                page_id,
                page_start,
                key[:, :, :take, :],
                value[:, :, :take, :],
            )
            self.page_lens[-1] += take
            start += take

        # Then allocate fresh pages for the remaining chunk. We only write the
        # valid prefix; unused tail slots are ignored by page_lens.
        while start < S:
            end = min(start + self.page_size, S)
            page_id = self.pool.allocate_page()
            self.pool.write_page_slice(
                page_id,
                0,
                key[:, :, start:end, :],
                value[:, :, start:end, :],
            )
            self.page_ids.append(page_id)
            self.page_lens.append(end - start)
            start = end

        self.offset += S

    def gather_dense(self) -> tuple[mx.array, mx.array]:
        assert self.offset > 0
        # Stage A compatibility path: attention still expects dense K/V, so we
        # trim each fixed-capacity page to its valid prefix and concatenate
        # request pages in logical order.
        key_chunks = []
        value_chunks = []
        for page_id, page_len in zip(self.page_ids, self.page_lens):
            key_page, value_page = self.pool.read_page(page_id)
            assert key_page.shape[2] == self.page_size
            assert value_page.shape[2] == self.page_size
            key_chunks.append(key_page[:, :, :page_len, :])
            value_chunks.append(value_page[:, :, :page_len, :])
        if len(key_chunks) == 1:
            return key_chunks[0], value_chunks[0]
        return mx.concat(key_chunks, axis=2), mx.concat(value_chunks, axis=2)

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        assert key.shape == value.shape
        self._append_chunk(key, value)
        # Day 1 keeps the old attention interface. Day 2 can replace this dense
        # gather with block_table/context_lens metadata.
        dense_key, dense_value = self.gather_dense()
        return dense_key, dense_value, self.offset, mask

    def rewind(self, n: int):
        assert 0 <= n <= self.offset
        new_offset = self.offset - n
        if new_offset == self.offset:
            return
        if new_offset == 0:
            self.release()
            return

        target_num_pages = (new_offset + self.page_size - 1) // self.page_size
        while len(self.page_ids) > target_num_pages:
            # Whole pages beyond the new logical length return to the shared
            # allocator. Stale suffix slots in the final page are ignored because
            # page_lens defines the valid prefix and future writes overwrite them.
            page_id = self.page_ids.pop()
            self.page_lens.pop()
            self.pool.free_page(page_id)

        last_page_len = new_offset - self.page_size * (target_num_pages - 1)
        self.page_lens[-1] = last_page_len
        self.offset = new_offset

    def release(self):
        # Request completion returns every page owned by this layer cache to the
        # model-level allocator. Other layer caches release their own pages.
        for page_id in self.page_ids:
            self.pool.free_page(page_id)
        self.page_ids.clear()
        self.page_lens.clear()
        self.offset = 0


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
        # Step 1: append each active row into its request cache. For paged
        # caches, this writes into the request's page table and then gathers a
        # dense view for the current Week 3 Day 1 attention path.
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

    def add_request(self, prefilled: TinyKvCache, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        if getattr(prefilled, "key_values", None) is not None:
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
