import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        # 1. top-k
        if top_k is not None and top_k > 0:
            mask_elements = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[:, top_k:]
            logprobs[:, mask_elements] = -mx.inf
        # 2. top-p
        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = logprobs[:, sorted_idx]
            cumsum = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)    # cumulative probs
            mask_elements = cumsum < top_p
            mask_elements[..., 0] = True  # always keep the first one
            logprobs[:, sorted_idx] = mx.where(mask_elements, sorted_logprobs, -mx.inf)
        # 3. temperature scaling
        logprobs = logprobs / temp
        # 4. sample
        return mx.random.categorical(logprobs, axis=-1)

    return sample
