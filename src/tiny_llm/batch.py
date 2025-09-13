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
        # TODO: in task 4, prefill the full request at once; in task 5, prefill a chunk at a time

    def decode_done(self, token, update_offset=True):
        if self.is_done:
            raise ValueError("decode called after done")
        if token == self.eos_token_id:
            self.is_done = True
            return
        # TODO: update the offset and add the token to the detokenizer

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
            text_preview = requests[i].text()[-80:].replace("\n", " ")
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
                # Implement this: find an idle slot and add the request to the decode requests
                pass
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
            # TODO: collect the next tokens and offsets from the decode requests
            next_tokens = _step(model, next_tokens.reshape(-1, 1), offsets, kv_cache)
            for i in range(batch_size):
                # TODO: check if the decode has finished by comparing EOS or the seqlength. If so,
                # remove the request from the decode requests and add the result to the result list;
                # otherwise, call `decode_done` to update the offset and add the token to the detokenizer
                pass
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
