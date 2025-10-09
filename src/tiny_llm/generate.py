import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    '''
    Simple greedy or sampling-based generation for Qwen2ModelWeek1.
    Args:
        model: Qwen2ModelWeek1 instance.
        tokenizer: TokenizerWrapper instance for encoding/decoding.
        prompt: Input prompt string to start generation.
        sampler: Optional callable that takes logprobs and returns sampled token ids.
     Returns:
         Generated string including the prompt and generated tokens.
    '''
    def _step(model, y):
        ''' Single generation step: given current tokens y, predict next token. '''
        logits = model(y[None]) # [batch, seq_len, vocab_size]
        logits = logits[:, -1, :]   # last token logits [batch, vocab_size]
        logprobs = logits - mx.logsumexp(logits, keepdims=True) # x - log(sum(exp(x))) for numerical stability
        if sampler is None:
            y = mx.argmax(logprobs, axis=-1)  # greedy
        else:
            y = sampler(logprobs)
        return y

    # 1. prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))  # [seq_len]
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    
    # 2. generate/decode
    while True:
        token = _step(model, tokens)  # [batch]
        mx.eval(token)  # ensure computation
        tokens = mx.concat([tokens, token])  # append new token
        if token.item() == tokenizer.eos_token_id:  # stop if EOS
            break
        detokenizer.add_token(token.item())  # add to detokenizer
        print(detokenizer.last_segment, end="", flush=True)  # print last segment

def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
