from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from .kv_cache import TinyKvCache


@dataclass
class PagedKvMetadata:
    key_pages: mx.array
    value_pages: mx.array
    block_table: mx.array
    context_lens: mx.array
    page_size: int
    mask: mx.array | str | None = None


class TinyKvPagedPool:
    """Model-local physical storage for paged KV.

    The model owns one pool and passes it to every layer cache. The pool gives
    out physical page ids from one free list. Because every live page id is
    unique, the page id alone is enough to find the physical K/V page.
    """

    def __init__(self, page_size: int = 128):
        assert page_size > 0
        self.page_size = page_size
        self.key_pages: mx.array | None = None
        self.value_pages: mx.array | None = None
        self.free_page_ids: list[int] = []
        self.used_page_ids: set[int] = set()
        self.num_allocated_pages = 0

    @property
    def num_pages(self) -> int:
        return self.num_allocated_pages

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
            self.num_allocated_pages += 1
        self.used_page_ids.add(page_id)
        return page_id

    def read_page(self, page_id: int) -> tuple[mx.array, mx.array]:
        if self.key_pages is None or self.value_pages is None:
            raise ValueError(f"Page {page_id} has no storage")
        if page_id >= self.num_pages:
            raise ValueError(f"Page {page_id} is out of range")
        return (
            self.key_pages[page_id : page_id + 1],
            self.value_pages[page_id : page_id + 1],
        )

    def _ensure_page_storage(self, key: mx.array, value: mx.array) -> None:
        B, H, _, D = key.shape
        assert B == 1

        if self.key_pages is not None and self.value_pages is not None:
            assert self.key_pages.shape[1:] == (H, self.page_size, D)
            assert self.value_pages.shape == self.key_pages.shape
            assert self.key_pages.dtype == key.dtype
            assert self.value_pages.dtype == value.dtype
            if self.key_pages.shape[0] >= self.num_pages:
                return

        new_key_pages = mx.zeros(
            (self.num_pages, H, self.page_size, D), dtype=key.dtype
        )
        new_value_pages = mx.zeros(
            (self.num_pages, H, self.page_size, D), dtype=value.dtype
        )
        if self.key_pages is not None and self.value_pages is not None:
            old_pages = self.key_pages.shape[0]
            new_key_pages[:old_pages, :, :, :] = self.key_pages
            new_value_pages[:old_pages, :, :, :] = self.value_pages
        self.key_pages = new_key_pages
        self.value_pages = new_value_pages

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
        self._ensure_page_storage(key, value)
        assert self.key_pages is not None
        assert self.value_pages is not None
        H, capacity, D = self.key_pages.shape[1:]
        assert self.value_pages.shape == self.key_pages.shape
        assert capacity == self.page_size
        assert key.shape[:2] == (1, H)
        assert key.shape[3] == D
        end = start + key.shape[2]
        assert 0 <= start <= capacity
        assert end <= self.page_size

        self.key_pages[page_id, :, start:end, :] = key[0]
        self.value_pages[page_id, :, start:end, :] = value[0]

    def free_page(self, page_id: int) -> None:
        if page_id not in self.used_page_ids:
            raise ValueError(f"Page {page_id} is already free")
        # Keep the page id stable. The stale K/V bytes can stay in the backing
        # tensor because block_table/page_lens decide which slots are live.
        self.used_page_ids.remove(page_id)
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
        # Dense compatibility path for tests and older callers. The paged
        # attention path uses block_table/context_lens instead of this gather.
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
        # Keep the old dense interface for Day 1 callers. Day 2 uses
        # update_and_fetch_paged so attention can read pages directly.
        dense_key, dense_value = self.gather_dense()
        return dense_key, dense_value, self.offset, mask

    def block_table(self, max_pages: int | None = None) -> mx.array:
        if max_pages is None:
            max_pages = self.num_pages
        assert max_pages >= self.num_pages
        page_ids = self.page_ids + [-1] * (max_pages - self.num_pages)
        return mx.array([page_ids], dtype=mx.int32)

    def context_lens(self) -> mx.array:
        return mx.array([self.offset], dtype=mx.int32)

    def paged_metadata(
        self,
        max_pages: int | None = None,
        mask: mx.array | str | None = None,
    ) -> PagedKvMetadata:
        assert self.pool.key_pages is not None
        assert self.pool.value_pages is not None
        return PagedKvMetadata(
            key_pages=self.pool.key_pages,
            value_pages=self.pool.value_pages,
            block_table=self.block_table(max_pages=max_pages),
            context_lens=self.context_lens(),
            page_size=self.page_size,
            mask=mask,
        )

    def update_and_fetch_paged(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> PagedKvMetadata:
        assert key.shape == value.shape
        self._append_chunk(key, value)
        return self.paged_metadata(mask=mask)

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
