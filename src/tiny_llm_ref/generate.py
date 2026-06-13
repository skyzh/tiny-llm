import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .kv_cache import *
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def _release_kv_cache(kv_cache):
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        logits = model(y[None])
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(
            logits, keepdims=True
        )  # optional -- for numerical stability
        if sampler is None:
            y = mx.argmax(logprobs, axis=-1)
        else:
            y = sampler(logprobs)
        return y

    # prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    # generate/decode
    while True:
        token = _step(model, tokens)
        mx.eval(token)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    kv_cache = model.create_kv_cache()

    def _step(model, y, offset, kv_cache):
        logits = model(y[None], offset, kv_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampler = lambda x: mx.argmax(x, axis=-1)
        y = sampler(logprobs)
        return y, logprobs.squeeze(0)

    try:
        # prefill with the prompt
        tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        offset = 0
        # generate/decode
        while True:
            token, _ = _step(model, tokens, offset, kv_cache)
            mx.eval(token)
            if token.item() == tokenizer.eos_token_id:
                break
            detokenizer.add_token(token.item())
            print(detokenizer.last_segment, end="", flush=True)
            # The first iteration of this loop is prefill. We want to add the offset to the prefilled token size.
            # Otherwise, we add the decoded token size (which is always 1).
            offset += tokens.size
            tokens = token
    finally:
        _release_kv_cache(kv_cache)


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    draft_kv_cache = draft_model.create_kv_cache()
    kv_cache = model.create_kv_cache()
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    def _step(model, y, offset, kv_cache, n_tokens=1):
        logits = model(y[None], offset, kv_cache)
        if n_tokens > 1:
            logits = logits[:, -n_tokens:, :]
        else:
            logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampler = lambda x: mx.argmax(x, axis=-1)
        y = sampler(logprobs)
        return y, logprobs.squeeze(0)

    def _prefill(model, tokenizer, prompt, kv_cache):
        prefill_tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
        token, _ = _step(model, prefill_tokens, 0, kv_cache)
        mx.eval(token)
        if token.item() == tokenizer.eos_token_id:
            return token, prefill_tokens.size
        return token, prefill_tokens.size

    def _decode_one(token):
        if token.item() == tokenizer.eos_token_id:
            return False
        detokenizer.add_token(token.item())
        return True

    def _draft_generate(model, last_token, offset, kv_cache, num_drafts):
        tokens = []
        current_offset = offset
        for _ in range(num_drafts):
            token, _ = _step(model, last_token, current_offset, kv_cache)
            mx.eval(token)
            tokens.append(token.item())
            last_token = token
            current_offset += 1
        return tokens

    def _rewind_cache(kv_cache, n_tokens):
        for layer in kv_cache:
            layer.rewind(n_tokens)

    def _print_progress(progress):
        newline = "\n"
        print(f"+{progress} {detokenizer.text.replace(newline, ' ')[-80:]}")

    try:
        _draft_token, draft_offset = _prefill(
            draft_model, draft_tokenizer, prompt, draft_kv_cache
        )
        token, offset = _prefill(model, tokenizer, prompt, kv_cache)
        if token.item() == tokenizer.eos_token_id:
            return detokenizer.text

        num_drafts = 4

        # speculative decode
        while True:
            draft_tokens = _draft_generate(
                draft_model, token, draft_offset, draft_kv_cache, num_drafts
            )
            draft_offset += num_drafts
            # assume both models use the same tokenizer
            draft_tokens = mx.concat([token, mx.array(draft_tokens)])
            new_tokens, _ = _step(model, draft_tokens, offset, kv_cache, num_drafts + 1)
            new_tokens = new_tokens.tolist()[0]
            offset += num_drafts + 1
            last_new_token = new_tokens[-1]
            new_tokens = mx.array([token.item()] + new_tokens[:-1])
            assert len(new_tokens) == len(draft_tokens)
            accept_all = True
            for i in range(len(new_tokens)):
                if new_tokens[i] != draft_tokens[i]:
                    # The accepted prefix is already emitted. Rewind both caches
                    # so the next round starts from the target replacement token.
                    assert i >= 1  # first token is always the same
                    revert_len = len(draft_tokens) - i
                    _rewind_cache(draft_kv_cache, revert_len - 1)
                    draft_offset -= revert_len - 1
                    _rewind_cache(kv_cache, revert_len)
                    token = mx.array([new_tokens[i]])
                    offset -= revert_len
                    assert offset == draft_offset
                    assert offset == kv_cache[0].offset
                    _print_progress(i)
                    accept_all = False
                    break
                if not _decode_one(new_tokens[i]):
                    print(detokenizer.text)
                    return detokenizer.text
            if accept_all:
                _print_progress(len(new_tokens))
                _draft_generate(
                    draft_model,
                    mx.array(draft_tokens[-1:]),
                    draft_offset,
                    draft_kv_cache,
                    1,
                )
                token = mx.array([last_new_token])
                draft_offset += 1
    finally:
        _release_kv_cache(draft_kv_cache)
        _release_kv_cache(kv_cache)
