"""Optional Week 3 speculative-decoding tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
import pytest

from .tiny_llm_base import speculative_generate


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
        # A deliberately persistent private detokenizer catches implementations
        # that reuse TokenizerWrapper internals across generation calls.
        self._detokenizer = FakeDetokenizer(self)

    def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
        assert prompt == "prompt"
        assert not add_special_tokens
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
    rewind_calls: list[int] = field(default_factory=list)
    release_count: int = 0

    def rewind(self, n: int):
        assert 0 < n <= self.offset
        self.offset -= n
        self.rewind_calls.append(n)

    def release(self):
        self.release_count += 1


class ScriptedModel:
    def __init__(self, outputs: list[list[int]], name: str):
        self.outputs = [list(output) for output in outputs]
        self.name = name
        self.calls: list[dict[str, object]] = []
        self.caches: list[FakeCache] = []

    def create_kv_cache(self) -> list[FakeCache]:
        cache = FakeCache()
        self.caches.append(cache)
        return [cache]

    def __call__(self, tokens, offset, kv_cache, logits_to_keep=1):
        cache = kv_cache[0]
        assert tokens.dtype == mx.int32
        assert offset == cache.offset
        token_ids = [int(token) for token in tokens.reshape(-1).tolist()]
        cache.offset += len(token_ids)

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
        assert all(cache.release_count == 1 for cache in model.caches)


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
    assert draft.calls == []
    assert draft.caches == []
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
    assert [call["tokens"] for call in target.calls] == [PROMPT, [1], [2]]
    assert all(call["dtype"] == mx.int32 for call in target.calls)
    assert draft.calls == []
    assert draft.caches == []
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
    assert [call["tokens"] for call in draft.calls] == [PROMPT]
    assert [call["tokens"] for call in target.calls] == [PROMPT, [1], [2]]
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
    assert target.caches[0].rewind_calls == [4 - mismatch_index]
    expected_draft_rewind = 3 - mismatch_index
    assert draft.caches[0].rewind_calls == (
        [expected_draft_rewind] if expected_draft_rewind else []
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
    assert [call["tokens"] for call in target.calls] == [PROMPT, [1, 2], [3, 4]]
    assert target.caches[0].rewind_calls == [1, 1]
    assert draft.caches[0].rewind_calls == []
    _assert_released(target, draft, target_only)
    assert target_only_draft.caches == []


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
    assert [call["tokens"] for call in target.calls] == [PROMPT, [1, 2, 3]]
    # Prefill plus two proposal calls: no draft catch-up may follow target EOS.
    assert [call["tokens"] for call in draft.calls] == [PROMPT, [1], [2]]
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
    assert [call["tokens"] for call in draft.calls] == [PROMPT, [1], [2]]
    assert target.caches[0].rewind_calls == [1]
    assert draft.caches[0].rewind_calls == []
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
    assert [call["offset"] for call in target.calls] == [0, 2, 5]
    assert [call["offset"] for call in draft.calls] == [0, 2, 3, 4, 5, 6]
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
    assert [call["tokens"] for call in draft.calls] == [PROMPT, [1], [3]]
    assert [call["logits_to_keep"] for call in target.calls] == [1, 2, 2]
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
    assert [item.reset_count for item in tokenizer.created_detokenizers] == [1, 1]
    assert tokenizer._detokenizer.tokens == []


@pytest.mark.parametrize(
    ("draft_tokenizer", "error"),
    [
        (FakeTokenizer([10, 12]), "encode the prompt differently"),
        (FakeTokenizer(eos_token_ids={EOS, 31}), "different EOS token ids"),
        (
            FakeTokenizer(vocab={"<eos>": EOS, "different": 1}),
            "different token ids",
        ),
    ],
)
def test_incompatible_tokenizers_fail_before_model_execution(
    draft_tokenizer: FakeTokenizer,
    error: str,
):
    target = ScriptedModel([], "target")
    draft = ScriptedModel([], "draft")

    with pytest.raises(ValueError, match=error):
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

    with pytest.raises(ValueError, match="comparable vocabularies"):
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

    with pytest.raises(ValueError, match="non-negative integer"):
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
