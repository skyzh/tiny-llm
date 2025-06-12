import torch

class RoPE:
    def __init__(self, head_dim: int, seq_len: int, base: int = 10000, traditional: bool = False):
        assert head_dim % 2 == 0, "head_dim must be even"
        half_dim = head_dim // 2

        theta = 1.0 / (base ** (torch.arange(0, half_dim).float() / half_dim))
        position = torch.arange(seq_len).float()
        freqs = torch.einsum("i,j->ij", position, theta)

        self.cos = freqs.cos()
        self.sin = freqs.sin()

        if traditional:
            self.cos = torch.repeat_interleave(self.cos, repeats=2, dim=-1)
            self.sin = torch.repeat_interleave(self.sin, repeats=2, dim=-1)

        self.traditional = traditional

    def __call__(self, x: torch.Tensor, offset: slice | None = None) -> torch.Tensor:
        bsz, seqlen, nheads, hd = x.shape
        assert hd % 2 == 0, "head_dim must be even"

        if offset is None:
            start = 0
            length = seqlen
        else:
            start = offset.start
            stop = offset.stop
            length = stop - start
            assert length == seqlen, f"Slice length {length} != input seq_len {seqlen}"

        if self.traditional:
            cos = self.cos[start: start + length].to(x.device)
            sin = self.sin[start: start + length].to(x.device)

            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)

            out = x * cos + self._rotate_half(x) * sin
            return out

        else:
            half_dim = hd // 2

            cos = self.cos[start: start + length].to(x.device)
            sin = self.sin[start: start + length].to(x.device)

            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)

            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]

            out1 = x1 * cos + x2 * sin
            out2 = -x1 * sin + x2 * cos

            out = torch.cat([out1, out2], dim=-1)
            return out

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
