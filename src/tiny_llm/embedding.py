import torch


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: torch.Tensor):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def as_linear(self, x: torch.Tensor) -> torch.Tensor:
        pass
