import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .kv_cache import *
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable
from datetime import datetime


def _step(model, y, offsets, kv_cache):
    logits = model(y, offsets, kv_cache)
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    sampler = lambda x: mx.argmax(x, axis=-1)
    y = sampler(logprobs)
    return y


class Request:
    def __init__(
        self,
        model: any,
        tokenizer: TokenizerWrapper,
        prompt: str,
        prefill_max_step: int = 128,
        prompt_idx: int = 0,
    ):
        self.prompt = prompt
        self.kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        self.model = model
        self.detokenizer = tokenizer.detokenizer.__class__(tokenizer._tokenizer)
        self.prefill_tokens = mx.array(
            tokenizer.encode(prompt, add_special_tokens=False)
        )
        self.prefill_max_step = prefill_max_step
        self.is_done = False
        self.is_prefill_done = False
        self.eos_token_id = tokenizer.eos_token_id
        self.next_token = None
        self.offset = 0
        self.prompt_idx = prompt_idx

    def try_prefill(self):
        """
        Prefill this request up to max_step size, returns None if prefill is not done
        """
        if self.is_prefill_done:
            raise ValueError("prefill called after done")
        tokens_to_prefill = min(
            self.prefill_max_step, self.prefill_tokens.size - self.offset
        )
        token = _step(
            self.model,
            self.prefill_tokens[self.offset : self.offset + tokens_to_prefill][None],
            [self.offset],
            self.kv_cache,
        )
        self.offset += tokens_to_prefill
        for i in self.kv_cache:
            mx.eval(i.key_values[0])
            mx.eval(i.key_values[1])
        if self.offset == self.prefill_tokens.size:
            self.is_prefill_done = True
            mx.eval(token)
            self.decode_done(token.item(), False)

    def decode_done(self, token, update_offset=True):
        if self.is_done:
            raise ValueError("decode called after done")
        if token == self.eos_token_id:
            self.is_done = True
            return
        self.detokenizer.add_token(token)
        self.next_token = token
        if update_offset:
            self.offset += 1

    def text(self):
        return self.detokenizer.text


def _print_progress(
    requests: list[Request | None],
    is_idle: list[bool],
    pending_prefill_request: Request | None,
    queue_size: int,
    progress_cnt: int,
    start_time: datetime,
):
    print(f"  --- {datetime.now() - start_time}")
    animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    animation_frame = animation_frames[progress_cnt % len(animation_frames)]
    for i in range(len(requests)):
        if is_idle[i]:
            print(f"  Decode #{i}: idle", flush=True)
        else:
            text_preview = requests[i].text()[-80:].replace('\n', ' ')
            print(
                f"{animation_frame} Decode [req {requests[i].prompt_idx}, {requests[i].offset}]: {text_preview}",
                flush=True,
            )
    if pending_prefill_request is not None:
        if pending_prefill_request.is_prefill_done:
            print(
                f"  Prefill [req {pending_prefill_request.prompt_idx}]: done, waiting for slot, {queue_size} requests in queue",
                flush=True,
            )
            return
        precentage = (
            pending_prefill_request.offset / pending_prefill_request.prefill_tokens.size
        ) * 100
        print(
            f"{animation_frame} Prefill [req {pending_prefill_request.prompt_idx}]: {precentage:.2f}% ({pending_prefill_request.prefill_tokens.size - pending_prefill_request.offset} remaining tokens)",
            flush=True,
        )
    else:
        print(f"  Prefill: idle, {queue_size} requests in queue", flush=True)


def batch_generate(
    model: any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    decode_requests: list[Request] = [None] * batch_size
    is_idle = [True] * batch_size
    kv_cache = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]
    result = []
    pending_prefill_request = None
    next_request_idx = 0
    progress_cnt = 0
    start_time = datetime.now()

    while True:
        if len(prompts) == 0 and all(is_idle):
            break
        # prefill until no idle slots
        if len(prompts) > 0 and pending_prefill_request is None:
            prompt = prompts.pop(0)
            pending_prefill_request = Request(
                model, tokenizer, prompt, prefill_step, next_request_idx
            )
            next_request_idx += 1

        # In every iteration, we do a prefill first
        if pending_prefill_request is not None:
            made_progress = False
            if not pending_prefill_request.is_prefill_done:
                pending_prefill_request.try_prefill()
                made_progress = True
            if pending_prefill_request.is_prefill_done:
                prefill_kv_cache = pending_prefill_request.kv_cache
                found_slot = False
                for i in range(batch_size):
                    if is_idle[i]:
                        # Add this request to the decode requests
                        is_idle[i] = False
                        for prefill_cache, batch_cache in zip(
                            prefill_kv_cache, kv_cache
                        ):
                            batch_cache.add_request(prefill_cache, i)
                        decode_requests[i] = pending_prefill_request
                        found_slot = True
                        made_progress = True
                        break
                if found_slot:
                    pending_prefill_request = None
            if made_progress:
                _print_progress(
                    decode_requests,
                    is_idle,
                    pending_prefill_request,
                    len(prompts),
                    progress_cnt,
                    start_time,
                )
                progress_cnt += 1

        # After the prefill request moves forward one step, we do the decode
        if not all(is_idle):
            next_tokens = []
            offsets = []
            for req in decode_requests:
                if req is not None:
                    next_tokens.append(req.next_token)
                    offsets.append(req.offset)
                else:
                    next_tokens.append(0)
                    offsets.append(0)
            next_tokens = mx.array(next_tokens)
            # decode
            next_tokens = _step(model, next_tokens.reshape(-1, 1), offsets, kv_cache)
            for i in range(batch_size):
                if not is_idle[i]:
                    req = decode_requests[i]
                    remove_reason = None
                    if req.is_done:
                        remove_reason = "EOS"
                    elif req.offset >= max_seq_len:
                        remove_reason = "max seq len"
                    if remove_reason is not None:
                        print(
                            f"Removing request {i} due to {remove_reason}", flush=True
                        )
                        batch_cache.remove_request(i)
                        is_idle[i] = True
                        result.append((req.prompt_idx, req.text()))
                        decode_requests[i] = None
                        continue
                    req.decode_done(next_tokens[i].item())
            _print_progress(
                decode_requests,
                is_idle,
                pending_prefill_request,
                len(prompts),
                progress_cnt,
                start_time,
            )
            progress_cnt += 1
    return result
