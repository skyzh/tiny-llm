import mlx.core as mx

from .tiny_llm_base import speculative_generate


EOS = 0
PROMPT = "prompt"
PROMPT_TOKENS = [1, 2]
PIECES = {
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    21: "X",
}


class FakeLayerCache:
    def __init__(self, name: str):
        self.name = name
        self.offset = 0
        self.updates = []
        self.rewinds = []
        self.release_count = 0
        self.released_offsets = []

    def append(self, offset: int, tokens: list[int]):
        assert self.offset == offset, (
            f"{self.name} expected offset {self.offset}, got {offset}"
        )
        self.updates.append((offset, tuple(tokens)))
        self.offset += len(tokens)

    def rewind(self, n: int):
        assert 0 <= n <= self.offset
        self.rewinds.append(n)
        self.offset -= n

    def release(self):
        self.release_count += 1
        self.released_offsets.append(self.offset)
        self.offset = 0


class ScriptedModel:
    def __init__(
        self,
        transitions: dict[int, int],
        name: str,
        vocab_size: int = 128,
    ):
        self.transitions = {EOS: EOS, **transitions}
        self.name = name
        self.vocab_size = vocab_size
        self.created_caches = []

    def create_kv_cache(self):
        cache = [FakeLayerCache(self.name)]
        self.created_caches.append(cache)
        return cache

    def __call__(self, inputs: mx.array, offset: int, cache: list[FakeLayerCache]):
        tokens = [int(token) for token in inputs.tolist()[0]]
        for layer in cache:
            layer.append(offset, tokens)

        logits = []
        for token in tokens:
            next_token = self.transitions.get(token, EOS)
            row = [-1000.0] * self.vocab_size
            row[next_token] = 1000.0
            logits.append(row)
        return mx.array([logits], dtype=mx.float32)


class FakeDetokenizer:
    def __init__(self, pieces: dict[int, str]):
        self.pieces = pieces
        self.reset()

    def reset(self):
        self.text = ""
        self.last_segment = ""

    def add_token(self, token: int):
        assert token != EOS, "EOS should stop generation, not be detokenized"
        self.last_segment = self.pieces[token]
        self.text += self.last_segment


class FakeTokenizer:
    eos_token_id = EOS

    def __init__(self, pieces: dict[int, str] | None = None):
        self._detokenizer = FakeDetokenizer(pieces or PIECES)

    @property
    def detokenizer(self):
        return self._detokenizer

    def encode(self, prompt: str, add_special_tokens: bool = False):
        assert prompt == PROMPT
        assert add_special_tokens is False
        return PROMPT_TOKENS


def run_speculative(
    target_transitions: dict[int, int],
    draft_transitions: dict[int, int],
):
    draft_model = ScriptedModel(draft_transitions, name="draft")
    target_model = ScriptedModel(target_transitions, name="target")
    draft_tokenizer = FakeTokenizer()
    tokenizer = FakeTokenizer()

    text = speculative_generate(
        draft_model,
        target_model,
        draft_tokenizer,
        tokenizer,
        PROMPT,
    )

    draft_layer = draft_model.created_caches[0][0]
    target_layer = target_model.created_caches[0][0]
    return text, draft_layer, target_layer


def test_task_1_accepts_full_draft_window_and_carries_extra_token():
    target = {
        2: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 14,
        14: 15,
        15: EOS,
    }
    draft = {
        2: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 14,
        14: 21,
        15: 21,
        21: 21,
    }

    text, draft_layer, target_layer = run_speculative(target, draft)

    assert text == "ABCDEF"
    assert target_layer.updates[1] == (2, (10, 11, 12, 13, 14))
    assert (6, (14,)) in draft_layer.updates
    assert draft_layer.release_count == 1
    assert target_layer.release_count == 1


def test_task_2_rejects_bad_suffix_and_rewinds_to_accepted_prefix():
    target = {
        2: 10,
        10: 11,
        11: 21,
        21: EOS,
    }
    draft = {
        2: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 14,
        21: 14,
        14: 14,
    }

    text, draft_layer, target_layer = run_speculative(target, draft)

    assert text == "ABX"
    assert draft_layer.rewinds[0] == 2
    assert target_layer.rewinds[0] == 3
    assert target_layer.updates[1] == (2, (10, 11, 12, 13, 14))


def test_task_2_rejects_at_first_draft_position():
    target = {
        2: 10,
        10: 21,
        21: EOS,
    }
    draft = {
        2: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 14,
        21: 14,
        14: 14,
    }

    text, draft_layer, target_layer = run_speculative(target, draft)

    assert text == "AX"
    assert draft_layer.rewinds[0] == 3
    assert target_layer.rewinds[0] == 4


def test_task_3_stops_on_eos_and_releases_caches():
    target = {
        2: 10,
        10: EOS,
    }
    draft = {
        2: 10,
        10: EOS,
    }

    text, draft_layer, target_layer = run_speculative(target, draft)

    assert text == "A"
    assert draft_layer.release_count == 1
    assert target_layer.release_count == 1
    assert draft_layer.rewinds == []
    assert target_layer.rewinds == []
