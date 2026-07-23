"""Week 2 Day 2 benchmark-lifecycle tests."""

import mlx.core as mx
import pytest

import bench


class TrackingCache:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class FakeModel:
    def __init__(self):
        self.caches = [TrackingCache(), TrackingCache()]

    def create_kv_cache(self):
        return self.caches


def test_single_request_benchmark_releases_cache(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(
        bench,
        "sample_next_week2",
        lambda model, tokens, offset, cache, logits_to_keep=1: mx.array(
            [7], dtype=mx.uint32
        ),
    )

    with mx.stream(mx.cpu):
        generated, _, _ = bench.run_one_request_week2(
            model,
            bench.BenchRequest(prompt_token_ids=[1, 2, 3], max_new_tokens=3),
        )

    assert generated == 3
    assert all(cache.released for cache in model.caches)


def test_single_request_benchmark_releases_cache_after_failure(monkeypatch):
    model = FakeModel()

    def fail(*args, **kwargs):
        raise RuntimeError("model failure")

    monkeypatch.setattr(bench, "sample_next_week2", fail)
    with pytest.raises(RuntimeError, match="model failure"):
        bench.run_one_request_week2(
            model,
            bench.BenchRequest(prompt_token_ids=[1], max_new_tokens=1),
        )

    assert all(cache.released for cache in model.caches)
