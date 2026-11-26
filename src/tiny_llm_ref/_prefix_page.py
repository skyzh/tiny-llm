"""Copy one completed prompt page into another request's own cache."""

from dataclasses import dataclass

import mlx.core as mx

from .paged_kv_cache import TinyKvPagedCache, TinyKvPagedPool
from .qwen3_week3 import Qwen3ModelWeek3


PAGE_SIZE = 128
LAYERS = 36
PAGE_SHAPE = (1, 8, PAGE_SIZE, 128)


@dataclass(frozen=True)
class _Snapshot:
    tokens: tuple[int, ...]
    layers: tuple[tuple[mx.array, mx.array], ...]


class _FirstPageCopy:
    def __init__(self, model):
        self.model = model
        self.snapshot = None
        self.captures = 0
        self.hits = 0

    @classmethod
    def for_queue(cls, model, prefill_step):
        if (
            type(model) is not Qwen3ModelWeek3
            or model.page_size != PAGE_SIZE
            or prefill_step != PAGE_SIZE
            or model.precision != mx.bfloat16
            or len(model.layers_inner) != LAYERS
            or len(model.page_pools) != LAYERS
        ):
            return None
        for layer, pool in zip(model.layers_inner, model.page_pools, strict=True):
            attn = layer.self_attn
            if (
                not attn.use_paged_attention
                or attn.num_kv_heads != PAGE_SHAPE[1]
                or attn.head_dim != PAGE_SHAPE[-1]
                or type(pool) is not TinyKvPagedPool
                or pool.page_size != PAGE_SIZE
                or pool.used_page_ids
            ):
                return None
        return cls(model)

    def _first_page_key(self, request):
        if (
            request.model is not self.model
            or request.offset != 0
            or request.is_prefill_done
            or request.prefill_max_step != PAGE_SIZE
            or request.prefill_tokens.ndim != 1
            or request.prefill_tokens.size <= PAGE_SIZE
            or request.prefill_tokens.dtype != mx.int32
            or len(request.kv_cache) != LAYERS
        ):
            return None
        for cache, pool in zip(request.kv_cache, self.model.page_pools, strict=True):
            if (
                type(cache) is not TinyKvPagedCache
                or cache.pool is not pool
                or cache.page_size != PAGE_SIZE
                or cache.offset != 0
                or cache.page_ids
                or cache.page_lens
            ):
                return None
        return tuple(request.prefill_tokens[:PAGE_SIZE].tolist())

    def _capture(self, tokens, request):
        if request.offset != PAGE_SIZE or request.is_prefill_done:
            return
        layers = []
        for cache in request.kv_cache:
            if (
                cache.offset != PAGE_SIZE
                or cache.page_lens != [PAGE_SIZE]
                or len(cache.page_ids) != 1
            ):
                return
            pair = cache.pool.read_page(cache.page_ids[0])
            if any(a.shape != PAGE_SHAPE or a.dtype != mx.bfloat16 for a in pair):
                return
            # Joining two halves materializes storage independent of the donor's pool.
            layers.append(
                tuple(
                    mx.concatenate(
                        (a[:, :, : PAGE_SIZE // 2], a[:, :, PAGE_SIZE // 2 :]), axis=2
                    )
                    for a in pair
                )
            )
        mx.eval([plane for pair in layers for plane in pair])
        self.snapshot = _Snapshot(tokens, tuple(layers))
        self.captures += 1

    def _restore(self, request, snapshot):
        caches = request.kv_cache
        states = [
            (cache, cache._snapshot_state(), cache.pool._snapshot_state())
            for cache in caches
        ]
        try:
            for cache, (key, value) in zip(caches, snapshot.layers, strict=True):
                cache._append_chunk(key, value)
            for cache in caches:
                cache.materialize()
        except BaseException:
            # Every attempted write belongs to the new request. Restore all
            # layer roots and page ownership before the scheduler releases it.
            for cache, cache_state, pool_state in reversed(states):
                cache.pool._restore_state(pool_state)
                cache._restore_state(cache_state)
            raise
        request.offset = PAGE_SIZE
        self.hits += 1

    def prefill(self, request):
        tokens = self._first_page_key(request)
        if tokens is not None and self.snapshot is not None:
            if tokens == self.snapshot.tokens:
                self._restore(request, self.snapshot)
                return
        request.try_prefill()
        if tokens is not None and self.snapshot is None:
            self._capture(tokens, request)

    def close(self):
        self.snapshot = None
