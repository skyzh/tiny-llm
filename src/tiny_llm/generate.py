import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable
from .kv_cache import TinyKvFullCache

def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        output_logits = model(y[None, :])
        logits = output_logits[:, -1, :]
        # avoid numerical instability
        if sampler is None:
            return mx.argmax(logits, axis=-1)
        else:
            logprobs = logits - mx.logsumexp(logits, keepdims=True) 
            return sampler(logprobs)


    context = mx.array(tokenizer.encode(prompt))
    token = _step(model, context)
    detokenizer = tokenizer.detokenizer
    while token.item() != tokenizer.eos_token_id:
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)

        context = mx.concat([context, token])
        token = _step(model, context)
        mx.eval(token)
    return ""
        


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        output_logits = model(y[None, :], offset, kv_cache)
        logits = output_logits[:, -1, :]
        # avoid numerical instability
        return mx.argmax(logits, axis=-1)

    offset = 0
    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]

    context = mx.array(tokenizer.encode(prompt))
    token = _step(model, context, offset, kv_cache)
    offset += context.size
    detokenizer = tokenizer.detokenizer
    while token.item() != tokenizer.eos_token_id:
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)

        token = _step(model, token, offset, kv_cache)
        offset += 1
        mx.eval(token)
    return ""


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
