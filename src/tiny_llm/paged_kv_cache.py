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
    def __init__(self, page_size: int = 128):
        pass

    @property
    def num_pages(self) -> int:
        pass

    @property
    def num_free_pages(self) -> int:
        pass

    def allocate_page(self) -> int:
        pass

    def read_page(self, page_id: int) -> tuple[mx.array, mx.array]:
        pass

    def write_page_slice(
        self,
        page_id: int,
        start: int,
        key: mx.array,
        value: mx.array,
    ) -> None:
        pass

    def free_page(self, page_id: int) -> None:
        pass


class TinyKvPagedCache(TinyKvCache):
    def __init__(self, pool: TinyKvPagedPool):
        pass

    @property
    def num_pages(self) -> int:
        pass

    def gather_dense(self) -> tuple[mx.array, mx.array]:
        pass

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        pass

    def block_table(self, max_pages: int | None = None) -> mx.array:
        pass

    def context_lens(self) -> mx.array:
        pass

    def paged_metadata(
        self,
        max_pages: int | None = None,
        mask: mx.array | str | None = None,
    ) -> PagedKvMetadata:
        pass

    def update_and_fetch_paged(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> PagedKvMetadata:
        pass

    def rewind(self, n: int):
        pass

    def release(self):
        pass
