import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .kv_cache import *
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def _release_kv_cache(kv_cache):
    if kv_cache is None:
        return
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> None:
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
        logits = model(y[None], offset, kv_cache, logits_to_keep=1)
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
    proposal_length: int = 4,
) -> str:
    if (
        not isinstance(proposal_length, int)
        or isinstance(proposal_length, bool)
        or proposal_length < 0
    ):
        raise ValueError("proposal_length must be a non-negative integer")

    def _encode(tokenizer):
        return [
            int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)
        ]

    def _eos_ids(tokenizer):
        eos_ids = getattr(tokenizer, "eos_token_ids", None)
        if eos_ids is None:
            eos_ids = {tokenizer.eos_token_id}
        return {int(token) for token in eos_ids}

    target_prompt_tokens = _encode(tokenizer)
    draft_prompt_tokens = _encode(draft_tokenizer)
    if not target_prompt_tokens:
        raise ValueError("prompt must encode to at least one token")
    if target_prompt_tokens != draft_prompt_tokens:
        raise ValueError("draft and target tokenizers encode the prompt differently")
    if _eos_ids(tokenizer) != _eos_ids(draft_tokenizer):
        raise ValueError("draft and target tokenizers use different EOS token ids")

    target_get_vocab = getattr(tokenizer, "get_vocab", None)
    draft_get_vocab = getattr(draft_tokenizer, "get_vocab", None)
    if not callable(target_get_vocab) or not callable(draft_get_vocab):
        raise ValueError(
            "draft and target tokenizers must expose comparable vocabularies"
        )
    if target_get_vocab() != draft_get_vocab():
        raise ValueError("draft and target tokenizers use different token ids")

    target_eos_ids = _eos_ids(tokenizer)
    draft_eos_ids = _eos_ids(draft_tokenizer)
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    kv_cache = model.create_kv_cache()
    draft_kv_cache = None

    def _step(model, y, offset, kv_cache, n_tokens=1):
        logits = model(y[None], offset, kv_cache, logits_to_keep=n_tokens)
        if n_tokens > 1:
            logits = logits[:, -n_tokens:, :]
        else:
            logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        y = mx.argmax(logprobs, axis=-1).astype(mx.int32)
        return y, logprobs.squeeze(0)

    def _token_array(tokens):
        return mx.array(tokens, dtype=mx.int32)

    def _token_id(token):
        return int(token.item())

    def _prefill(model, prefill_tokens, kv_cache):
        token, _ = _step(model, _token_array(prefill_tokens), 0, kv_cache)
        mx.eval(token)
        return _token_id(token), len(prefill_tokens)

    def _rewind_cache(kv_cache, revert_len):
        if revert_len == 0:
            return
        for layer in kv_cache:
            layer.rewind(revert_len)

    def _assert_cache_offset(kv_cache, expected):
        for layer in kv_cache:
            if hasattr(layer, "offset"):
                assert layer.offset == expected

    def _print_text(text, progress):
        newline = "\n"
        print(f"+{progress} {text.replace(newline, ' ')[-80:]}")

    def _emit(token_ids):
        for token_id in token_ids:
            detokenizer.add_token(token_id)
        if token_ids:
            _print_text(detokenizer.text, len(token_ids))

    def _finish():
        finalize = getattr(detokenizer, "finalize", None)
        if callable(finalize):
            finalize()
        text = detokenizer.text
        print(text)
        return text

    def _target_only(token_id, offset):
        while True:
            if token_id in target_eos_ids:
                return _finish()
            _emit([token_id])
            token, _ = _step(
                model,
                _token_array([token_id]),
                offset,
                kv_cache,
            )
            mx.eval(token)
            offset += 1
            _assert_cache_offset(kv_cache, offset)
            token_id = _token_id(token)

    try:
        token_id, offset = _prefill(model, target_prompt_tokens, kv_cache)
        _assert_cache_offset(kv_cache, offset)
        if token_id in target_eos_ids:
            return _finish()
        if proposal_length == 0:
            return _target_only(token_id, offset)

        draft_kv_cache = draft_model.create_kv_cache()
        draft_token_id, draft_offset = _prefill(
            draft_model,
            draft_prompt_tokens,
            draft_kv_cache,
        )
        _assert_cache_offset(draft_kv_cache, draft_offset)
        assert offset == draft_offset
        if draft_token_id in draft_eos_ids:
            return _target_only(token_id, offset)

        def _draft_generate(last_token_id, offset, max_tokens):
            tokens: list[int] = []
            current_offset = offset
            for _ in range(max_tokens):
                token, _ = _step(
                    draft_model,
                    _token_array([last_token_id]),
                    current_offset,
                    draft_kv_cache,
                )
                mx.eval(token)
                last_token_id = _token_id(token)
                tokens.append(last_token_id)
                current_offset += 1
                if last_token_id in draft_eos_ids:
                    break
            return tokens, current_offset

        while True:
            draft_tokens, draft_offset = _draft_generate(
                token_id,
                draft_offset,
                proposal_length,
            )
            _assert_cache_offset(draft_kv_cache, draft_offset)

            verification_ids = [token_id, *draft_tokens]
            new_tokens, _ = _step(
                model,
                _token_array(verification_ids),
                offset,
                kv_cache,
                len(verification_ids),
            )
            mx.eval(new_tokens)
            target_predictions = [
                int(value) for value in new_tokens.reshape(-1).tolist()
            ]
            assert len(target_predictions) == len(verification_ids)
            offset += len(verification_ids)
            _assert_cache_offset(kv_cache, offset)

            aligned_target = [token_id, *target_predictions[:-1]]
            mismatch_index = None
            terminal_index = None
            for i, (target_id, draft_id) in enumerate(
                zip(aligned_target, verification_ids, strict=True)
            ):
                if target_id != draft_id:
                    mismatch_index = i
                    break
                if target_id in target_eos_ids:
                    terminal_index = i
                    break

            if terminal_index is not None:
                _emit(aligned_target[:terminal_index])
                target_rewind = len(verification_ids) - terminal_index
                draft_rewind = len(draft_tokens) - terminal_index
                _rewind_cache(kv_cache, target_rewind)
                _rewind_cache(draft_kv_cache, draft_rewind)
                offset -= target_rewind
                draft_offset -= draft_rewind
                assert offset == draft_offset
                _assert_cache_offset(kv_cache, offset)
                _assert_cache_offset(draft_kv_cache, draft_offset)
                return _finish()

            if mismatch_index is not None:
                assert mismatch_index >= 1
                _emit(aligned_target[:mismatch_index])
                target_rewind = len(verification_ids) - mismatch_index
                draft_rewind = len(draft_tokens) - mismatch_index
                _rewind_cache(kv_cache, target_rewind)
                _rewind_cache(draft_kv_cache, draft_rewind)
                offset -= target_rewind
                draft_offset -= draft_rewind
                assert offset == draft_offset
                _assert_cache_offset(kv_cache, offset)
                _assert_cache_offset(draft_kv_cache, draft_offset)
                token_id = aligned_target[mismatch_index]
                if token_id in target_eos_ids:
                    return _finish()
                continue

            _emit(aligned_target)
            bonus_token_id = target_predictions[-1]
            if bonus_token_id in target_eos_ids:
                return _finish()

            _, draft_offset = _draft_generate(
                verification_ids[-1],
                draft_offset,
                1,
            )
            token_id = bonus_token_id
            assert offset == draft_offset
            _assert_cache_offset(kv_cache, offset)
            _assert_cache_offset(draft_kv_cache, draft_offset)
    finally:
        _release_kv_cache(draft_kv_cache)
        _release_kv_cache(kv_cache)
