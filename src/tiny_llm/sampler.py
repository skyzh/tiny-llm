import torch


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: torch.Tensor):
        if temp == 0:
            return torch.argmax(logprobs, dim=-1)
        pass

    return sample
