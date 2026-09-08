"""Optional Week 3 speculative-decoding tests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

import mlx.core as mx
import numpy as np
import pytest

from .tiny_llm_base import (
    TinyKvFullCache,
    TinyKvPagedCache,
    TinyKvPagedPool,
    speculative_generate,
)


EOS = 0
PROMPT = [10, 11]
PIECES = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
}


class FakeDetokenizer:
    def __init__(self, tokenizer: "FakeTokenizer"):
        self.tokenizer = tokenizer
        self.tokens: list[int] = []
        self.offset = 0
        self.reset_count = 0
        self.finalize_count = 0

    def reset(self):
        self.tokens = []
        self.offset = 0
        self.reset_count += 1

    def add_token(self, token: int):
        self.tokens.append(token)

    def finalize(self):
        self.finalize_count += 1

    @property
    def text(self) -> str:
        return "".join(self.tokenizer.pieces[token] for token in self.tokens)

    @property
    def last_segment(self) -> str:
        text = self.text
        segment = text[self.offset :]
        self.offset = len(text)
        return segment


class FakeTokenizer:
    def __init__(
        self,
        prompt_tokens: list[int] | None = None,
        *,
        eos_token_ids: set[int] | None = None,
        vocab: dict[str, int] | None = None,
    ):
        self.prompt_tokens = list(prompt_tokens or PROMPT)
        self.eos_token_id = EOS
        self.eos_token_ids = set(eos_token_ids or {EOS})
        self.pieces = PIECES
        self.vocab = vocab or {
            "<eos>": EOS,
            **{text: token for token, text in PIECES.items()},
            "prompt-a": PROMPT[0],
            "prompt-b": PROMPT[1],
        }
        self.created_detokenizers: list[FakeDetokenizer] = []
        self.encode_count = 0

    def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
        assert prompt == "prompt"
        assert not add_special_tokens
        self.encode_count += 1
        return list(self.prompt_tokens)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    @property
    def detokenizer(self) -> FakeDetokenizer:
        detokenizer = FakeDetokenizer(self)
        self.created_detokenizers.append(detokenizer)
        return detokenizer


@dataclass
class FakeCache:
    offset: int = 0
    released: bool = False

    def rewind(self, n: int):
        assert 0 < n <= self.offset
        self.offset -= n

    def release(self):
        self.released = True
        self.offset = 0


class ScriptedModel:
    def __init__(
        self,
        outputs: list[list[int]],
        name: str,
        cache_factory: Callable[[], object] = FakeCache,
    ):
        self.outputs = [list(output) for output in outputs]
        self.name = name
        self.calls: list[dict[str, object]] = []
        self.caches: list[object] = []
        self.cache_factory = cache_factory

    def create_kv_cache(self) -> list[object]:
        cache = self.cache_factory()
        self.caches.append(cache)
        return [cache]

    def __call__(self, tokens, offset, kv_cache, logits_to_keep=1):
        cache = kv_cache[0]
        assert tokens.dtype == mx.int32
        assert offset == cache.offset
        token_ids = [int(token) for token in tokens.reshape(-1).tolist()]
        if isinstance(cache, FakeCache):
            cache.offset += len(token_ids)
        else:
            length = len(token_ids)
            values = mx.zeros((1, 1, length, 4), dtype=mx.float32)
            cache.update_and_fetch(values, values)

        if not self.outputs:
            raise AssertionError(f"{self.name} received an unexpected model call")
        output = self.outputs.pop(0)
        assert len(output) == logits_to_keep
        self.calls.append(
            {
                "tokens": token_ids,
                "offset": offset,
                "logits_to_keep": logits_to_keep,
                "dtype": tokens.dtype,
            }
        )

        logits = np.full((1, len(output), 32), -1000.0, dtype=np.float32)
        for position, token_id in enumerate(output):
            logits[0, position, token_id] = 1000.0
        return mx.array(logits)


def _tokenizers() -> tuple[FakeTokenizer, FakeTokenizer]:
    return FakeTokenizer(), FakeTokenizer()


def _assert_released(*models: ScriptedModel):
    for model in models:
        assert all(cache.released for cache in model.caches)


def test_target_prefill_eos_finishes_before_the_draft_runs():
    target = ScriptedModel([[EOS]], "target")
    draft = ScriptedModel([], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
    )

    assert result == ""
    assert len(target.calls) == 1
    assert not draft.calls
    assert not draft.caches
    _assert_released(target)


def test_zero_proposal_length_is_target_only_and_uses_int32_tokens():
    target = ScriptedModel([[1], [2], [EOS]], "target")
    draft = ScriptedModel([], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=0,
    )

    assert result == "AB"
    assert len(target.calls) == 3
    assert all(call["dtype"] == mx.int32 for call in target.calls)
    assert not draft.calls
    assert not draft.caches
    _assert_released(target)


def test_draft_prefill_eos_falls_back_to_target_only():
    target = ScriptedModel([[1], [2], [EOS]], "target")
    draft = ScriptedModel([[EOS]], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
    )

    assert result == "AB"
    assert len(draft.calls) == 1
    assert len(target.calls) == 3
    _assert_released(target, draft)


@pytest.mark.parametrize("mismatch_index", [1, 2, 3])
def test_mismatch_rewinds_first_middle_and_final_proposal(mismatch_index: int):
    proposal = [2, 3, 4]
    predictions = [7, 7, 7, 7]
    predictions[: mismatch_index - 1] = proposal[: mismatch_index - 1]
    predictions[mismatch_index - 1] = EOS

    target = ScriptedModel([[1], predictions], "target")
    draft = ScriptedModel([[9], [2], [3], [4]], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=3,
    )

    assert result == "".join(
        PIECES[token] for token in [1, *proposal][0:mismatch_index]
    )
    _assert_released(target, draft)


def test_low_acceptance_mismatches_match_the_complete_target_only_output():
    target = ScriptedModel(
        [
            [1],
            [3, 7],
            [EOS, 7],
        ],
        "target",
    )
    draft = ScriptedModel([[9], [2], [4]], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    speculative_result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=1,
    )

    target_only = ScriptedModel([[1], [3], [EOS]], "target-only")
    target_only_draft = ScriptedModel([], "unused-draft")
    target_only_draft_tokenizer, target_only_tokenizer = _tokenizers()
    target_only_result = speculative_generate(
        target_only_draft,
        target_only,
        target_only_draft_tokenizer,
        target_only_tokenizer,
        "prompt",
        proposal_length=0,
    )

    assert speculative_result == target_only_result == "AC"
    assert sum(call["logits_to_keep"] > 1 for call in target.calls) == 2
    _assert_released(target, draft, target_only)
    assert not target_only_draft.caches


def test_full_acceptance_stops_on_bonus_eos_without_a_followup_model_call():
    target = ScriptedModel([[1], [2, 3, EOS]], "target")
    draft = ScriptedModel([[9], [2], [3]], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=2,
    )

    assert result == "ABC"
    assert len(target.calls) == 2
    # Prefill plus two proposal calls: no call may follow target EOS.
    assert len(draft.calls) == 3
    _assert_released(target, draft)


def test_matching_eos_inside_a_short_proposal_is_terminal():
    target = ScriptedModel([[1], [2, EOS, 7]], "target")
    draft = ScriptedModel([[9], [2], [EOS]], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=4,
    )

    assert result == "AB"
    assert len(target.calls) == 2
    assert len(draft.calls) == 3
    _assert_released(target, draft)


def test_full_acceptance_catches_up_before_the_next_proposal():
    target = ScriptedModel(
        [
            [1],
            [2, 3, 4],
            [5, 6, EOS],
        ],
        "target",
    )
    draft = ScriptedModel(
        [
            [9],
            [2],
            [3],
            [9],
            [5],
            [6],
        ],
        "draft",
    )
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=2,
    )

    assert result == "ABCDEF"
    assert len(target.calls) == 3
    assert len(draft.calls) == 6
    _assert_released(target, draft)


def test_draft_proposal_stops_early_at_eos_without_terminating_target():
    target = ScriptedModel(
        [
            [1],
            [3, 7],
            [EOS, 7],
        ],
        "target",
    )
    draft = ScriptedModel([[9], [EOS], [EOS]], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=4,
    )

    assert result == "AC"
    assert len(draft.calls) == 3
    assert [call["logits_to_keep"] for call in target.calls[1:]] == [2, 2]
    _assert_released(target, draft)


def test_repeated_calls_use_fresh_public_detokenizers():
    draft_tokenizer, tokenizer = _tokenizers()

    results = []
    for _ in range(2):
        target = ScriptedModel([[1], [2], [EOS]], "target")
        draft = ScriptedModel([], "draft")
        results.append(
            speculative_generate(
                draft,
                target,
                draft_tokenizer,
                tokenizer,
                "prompt",
                proposal_length=0,
            )
        )
        _assert_released(target)

    assert results == ["AB", "AB"]
    assert len(tokenizer.created_detokenizers) == 2
    assert all(item.reset_count == 1 for item in tokenizer.created_detokenizers)


@pytest.mark.parametrize(
    "draft_tokenizer",
    [
        FakeTokenizer([10, 12]),
        FakeTokenizer(eos_token_ids={EOS, 31}),
        FakeTokenizer(vocab={"<eos>": EOS, "different": 1}),
    ],
)
def test_incompatible_tokenizers_fail_before_model_execution(
    draft_tokenizer: FakeTokenizer,
):
    target = ScriptedModel([], "target")
    draft = ScriptedModel([], "draft")

    with pytest.raises(ValueError):
        speculative_generate(
            draft,
            target,
            draft_tokenizer,
            FakeTokenizer(),
            "prompt",
        )

    assert target.caches == []
    assert draft.caches == []


def test_tokenizers_without_comparable_vocabularies_fail_before_execution():
    target = ScriptedModel([], "target")
    draft = ScriptedModel([], "draft")
    draft_tokenizer, tokenizer = _tokenizers()
    draft_tokenizer.get_vocab = None

    with pytest.raises(ValueError):
        speculative_generate(
            draft,
            target,
            draft_tokenizer,
            tokenizer,
            "prompt",
        )

    assert target.caches == []
    assert draft.caches == []


@pytest.mark.parametrize("proposal_length", [-1, 1.5, "4", True])
def test_invalid_proposal_length_fails_before_model_execution(proposal_length):
    target = ScriptedModel([], "target")
    draft = ScriptedModel([], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    with pytest.raises(ValueError):
        speculative_generate(
            draft,
            target,
            draft_tokenizer,
            tokenizer,
            "prompt",
            proposal_length=proposal_length,
        )

    assert target.caches == []
    assert draft.caches == []


@pytest.mark.parametrize("max_tokens", [-1, 1.5, "4", True])
def test_invalid_output_budget_fails_before_model_or_tokenizer_execution(max_tokens):
    target = ScriptedModel([], "target")
    draft = ScriptedModel([], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    with pytest.raises(ValueError):
        speculative_generate(
            draft,
            target,
            draft_tokenizer,
            tokenizer,
            "prompt",
            max_tokens=max_tokens,
        )

    assert not target.caches
    assert not draft.caches
    assert tokenizer.encode_count == 0
    assert draft_tokenizer.encode_count == 0


def test_zero_output_budget_returns_without_model_cache_or_tokenizer_work():
    target = ScriptedModel([], "target")
    draft = ScriptedModel([], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        max_tokens=0,
    )

    assert result == ""
    assert not target.caches
    assert not draft.caches
    assert tokenizer.encode_count == 0
    assert draft_tokenizer.encode_count == 0


def test_target_only_budget_stops_before_eos_without_an_extra_model_call():
    target = ScriptedModel([[1], [2]], "target")
    draft = ScriptedModel([], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=0,
        max_tokens=2,
    )

    assert result == "AB"
    assert len(target.calls) == 2
    assert not draft.caches
    _assert_released(target)


def test_target_only_eos_stops_before_the_output_budget():
    target = ScriptedModel([[1], [EOS]], "target")
    draft = ScriptedModel([], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=0,
        max_tokens=8,
    )

    assert result == "A"
    assert len(target.calls) == 2
    _assert_released(target)


def test_full_acceptance_honors_output_budget_without_catch_up():
    target = ScriptedModel([[1], [2, 3]], "target")
    draft = ScriptedModel([[9], [2]], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=4,
        max_tokens=2,
    )

    assert result == "AB"
    assert len(target.calls) == 2
    assert len(draft.calls) == 2
    _assert_released(target, draft)


def test_full_acceptance_emits_the_last_bonus_without_a_stale_draft_call():
    target = ScriptedModel([[1], [2, 3]], "target")
    draft = ScriptedModel([[9], [2]], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=1,
        max_tokens=3,
    )

    assert result == "ABC"
    assert len(target.calls) == 2
    assert len(draft.calls) == 2
    _assert_released(target, draft)


@pytest.mark.parametrize("cache_kind", ["dense", "paged"])
@pytest.mark.parametrize("scenario", ["mismatch", "full_acceptance"])
def test_real_cache_paths_preserve_output_and_release_pages(cache_kind, scenario):
    pools: list[TinyKvPagedPool] = []

    def cache_factory():
        if cache_kind == "dense":
            return TinyKvFullCache()
        pool = TinyKvPagedPool(page_size=2)
        pools.append(pool)
        return TinyKvPagedCache(pool=pool)

    if scenario == "mismatch":
        target_outputs = [[1], [3, 7]]
        draft_outputs = [[9], [2]]
        expected = "AC"
        proposal_length = 1
    else:
        target_outputs = [[1], [2, 3]]
        draft_outputs = [[9], [2]]
        expected = "AB"
        proposal_length = 2

    target = ScriptedModel(target_outputs, "target", cache_factory)
    draft = ScriptedModel(draft_outputs, "draft", cache_factory)
    draft_tokenizer, tokenizer = _tokenizers()

    result = speculative_generate(
        draft,
        target,
        draft_tokenizer,
        tokenizer,
        "prompt",
        proposal_length=proposal_length,
        max_tokens=2,
    )

    assert result == expected
    if cache_kind == "paged":
        assert pools
        assert all(pool.num_free_pages == pool.num_pages for pool in pools)


def test_model_failure_releases_created_caches():
    target = ScriptedModel([[1]], "target")
    draft = ScriptedModel([[9], [2]], "draft")
    draft_tokenizer, tokenizer = _tokenizers()

    with pytest.raises(AssertionError):
        speculative_generate(
            draft,
            target,
            draft_tokenizer,
            tokenizer,
            "prompt",
            proposal_length=1,
        )

    _assert_released(target, draft)


def _run_main(*args: str) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "sampler_args",
    [
        ("--sampler-temp", "0.5"),
        ("--sampler-top-p", "0.9"),
        ("--sampler-top-k", "8"),
    ],
)
def test_cli_rejects_sampled_draft_model_before_loading_models(sampler_args):
    result = _run_main(
        "--solution",
        "ref",
        "--loader",
        "week3",
        "--model",
        "model-must-not-load",
        "--draft-model",
        "draft-must-not-load",
        "--max-tokens",
        "0",
        *sampler_args,
    )

    assert result.returncode != 0


def test_cli_zero_output_budget_returns_before_loading_models():
    result = _run_main(
        "--solution",
        "ref",
        "--loader",
        "week3",
        "--model",
        "model-must-not-load",
        "--draft-model",
        "draft-must-not-load",
        "--max-tokens",
        "0",
    )

    assert result.returncode == 0


def test_cli_rejects_negative_output_budget_before_loading_models():
    result = _run_main(
        "--solution",
        "ref",
        "--loader",
        "week3",
        "--model",
        "model-must-not-load",
        "--max-tokens",
        "-1",
    )

    assert result.returncode != 0
