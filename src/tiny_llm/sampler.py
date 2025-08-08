import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        pass

    return sample
