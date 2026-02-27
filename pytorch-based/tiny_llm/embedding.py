import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, weight: torch.Tensor = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if weight is not None:
            self.embedding.weight.data = weight.clone().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def as_linear(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.weight)
