from dataclasses import dataclass
from typing import Optional

import torch

from .kv_cache import TinyKvCache


@dataclass
class PagedKvMetadata:
    key_pages: torch.Tensor
    value_pages: torch.Tensor
    block_table: torch.Tensor
    context_lens: torch.Tensor
    page_size: int
    mask: torch.Tensor | str | None = None


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

    def read_page(self, page_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def write_page_slice(
        self,
        page_id: int,
        start: int,
        key: torch.Tensor,
        value: torch.Tensor,
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

    def gather_dense(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def update_and_fetch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        pass

    def block_table(self, max_pages: int | None = None) -> torch.Tensor:
        pass

    def context_lens(self) -> torch.Tensor:
        pass

    def paged_metadata(
        self,
        max_pages: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> PagedKvMetadata:
        pass

    def update_and_fetch_paged(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> PagedKvMetadata:
        pass

    def rewind(self, n: int):
        pass

    def release(self):
        pass
