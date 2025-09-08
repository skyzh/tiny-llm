import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from .basics import logsumexp_norm
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        output_logits = model(y[None])
        logits = output_logits[:, -1, :]
        logprobs = logsumexp_norm(logits)
        if sampler is None:
            return mx.argmax(logprobs, axis=-1)
        else:
            return sampler(logprobs)
    
    prompt_tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    next_token = None
    
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    
    while next_token is None or next_token.item() != tokenizer.eos_token_id:
        next_token = _step(model, prompt_tokens)
        mx.eval(next_token)
        prompt_tokens = mx.concat([prompt_tokens, next_token])
        if next_token.item() != tokenizer.eos_token_id:
            detokenizer.add_token(next_token.item())
            print(detokenizer.last_segment, end="", flush=True)


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def batch_generate(
    model: any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    pass
