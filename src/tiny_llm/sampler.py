import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if top_k is not None and top_k > 0:
            mask_idx = mx.argpartition(-logprobs, top_k - 1)[..., top_k:]
            logprobs = mx.put_along_axis(logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1)
        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = logprobs[..., sorted_idx]
            cumsum = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)
            mask = cumsum < top_p
            mask[..., 0] = True # ensure at least one token is kept
            logprobs[..., sorted_idx] = mx.where(mask, sorted_logprobs, -mx.inf)

        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        else:
            return mx.random.categorical(logprobs / temp, axis=-1)

    return sample

