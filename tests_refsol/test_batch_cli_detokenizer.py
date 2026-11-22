"""Check the reference CLI shortcut with a small mutable vocabulary."""

from types import SimpleNamespace

from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

from tiny_llm_ref import batch


class TinyTokenizer:
    eos_token_id = 0
    chat_template = None
    clean_up_tokenization_spaces = False

    def __init__(self):
        # The last two entries are the two BPE-encoded bytes of é.
        self.vocab = {"[EOS]": 0, "x": 1, "Ġ": 2, "Ã": 3, "©": 4}

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def convert_ids_to_tokens(self, ids):
        by_id = {index: token for token, index in self.vocab.items()}
        return [by_id[index] for index in ids]

    def apply_chat_template(self, *args, **kwargs):
        return "x"

    def encode(self, prompt, *, add_special_tokens):
        return [self.vocab[prompt]]

    def add_tokens(self, tokens):
        for token in tokens:
            self.vocab.setdefault(token, len(self.vocab))


def test_cli_detokenizer_keeps_independent_request_state(monkeypatch):
    tokenizer = TokenizerWrapper(TinyTokenizer(), BPEStreamingDetokenizer)
    model = SimpleNamespace(create_kv_cache=list)
    constructor = BPEStreamingDetokenizer.__init__
    constructed = []

    def counted_init(self, tokenizer):
        constructed.append(self)
        constructor(self, tokenizer)

    monkeypatch.setattr(BPEStreamingDetokenizer, "__init__", counted_init)
    cli = [batch._CliRequest(model, tokenizer, "x") for _ in range(2)]
    cli_constructions = len(constructed)
    public = [batch.Request(model, tokenizer, "x") for _ in range(2)]
    assert len(constructed) - cli_constructions == cli_constructions + len(cli)
    assert cli[0].detokenizer is not cli[1].detokenizer
    assert cli[0].detokenizer.tokens is not cli[1].detokenizer.tokens
    lookup = list(cli[0].detokenizer.tokenmap)

    events = [(0, 2, ""), (1, 1, "x"), (0, 3, ""), (0, 4, "é"), (1, 0, "x"), (0, 0, "é")]
    for slot, token, expected in events:
        cli[slot].decode_done(token)
        public[slot].decode_done(token)
        assert cli[slot].text() == public[slot].text() == expected
        assert cli[slot].detokenizer.tokens == public[slot].detokenizer.tokens
        assert cli[slot].finish_reason == public[slot].finish_reason

    assert all(request.is_done for request in cli)
    assert cli[0].detokenizer.tokenmap == cli[1].detokenizer.tokenmap == lookup


def test_batch_detokenizer_decodes_a_new_token_after_wrapper_creation(monkeypatch):
    tokenizer = TokenizerWrapper(TinyTokenizer(), BPEStreamingDetokenizer)
    tokenizer.add_tokens(["z"])
    added_id = tokenizer.get_vocab()["z"]
    monkeypatch.setattr(batch, "_step", lambda *args: batch.mx.array(added_id))
    model = SimpleNamespace(num_hidden_layers=0, create_kv_cache=list)

    for generate in (batch.batch_generate, batch._batch_generate_with_cli_tokenizer):
        assert generate(
            model, tokenizer, ["x"], max_seq_len=2, batch_size=1, prefill_step=1
        ) == [(0, "z")]


def test_cli_detokenizer_uses_the_reported_class_for_custom_wrappers():
    raw_tokenizer = object()

    class Detokenizer:
        def __init__(self, tokenizer):
            assert tokenizer is raw_tokenizer

    class Proxy:
        @property
        def __class__(self):
            return Detokenizer

    wrapper = TinyTokenizer()
    wrapper._tokenizer = raw_tokenizer
    wrapper.detokenizer = Proxy()
    model = SimpleNamespace(create_kv_cache=list)

    for request_type in (batch.Request, batch._CliRequest):
        assert type(request_type(model, wrapper, "x").detokenizer) is Detokenizer


def test_cli_detokenizer_preserves_custom_wrapper_handling():
    raw_tokenizer = TinyTokenizer()
    instance = BPEStreamingDetokenizer(raw_tokenizer)

    class CustomWrapper(TokenizerWrapper):
        @property
        def detokenizer(self):
            return instance

    wrapper = CustomWrapper(raw_tokenizer, BPEStreamingDetokenizer)
    request = batch._CliRequest(SimpleNamespace(create_kv_cache=list), wrapper, "x")
    assert type(request.detokenizer) is BPEStreamingDetokenizer
    assert request.detokenizer is not instance
