import mlx.core as mx
import torch

with mx.stream(mx.cpu):
    a = mx.array([1, 2, 3])
    b = mx.array([4, 5, 6])
    c = mx.add(a, b)
    print(c)

with mx.stream(mx.gpu):
    a = mx.array([1, 2, 3])
    b = mx.array([4, 5, 6])
    c = mx.add(a, b)
    print(c)

print(
    torch.add(
        torch.tensor([1, 2, 3], device="cpu"), torch.tensor([4, 5, 6], device="cpu")
    )
)
