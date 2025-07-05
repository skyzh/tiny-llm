import mlx.core as mx
from .basics import softmax, linear

"""
key: N x H x L x D
value: N x H x L x D
query: N x H x L x D
output: N x H x L x D
mask: N x H x L x L
"""
def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    D = query.shape[-1]

    # For multi-dimensional arrays, transpose only the last two dimensions
    key_transposed = mx.swapaxes(key, -2, -1)
    qk = mx.matmul(query, key_transposed)
    scale = 1 / mx.sqrt(D) if scale is None else scale

    # Apply scale
    scaled_qk = qk * scale
    
    # Add mask if provided
    if mask is not None:
        scaled_qk = scaled_qk + mask
    
    return mx.softmax(scaled_qk, axis=-1) @ value

"""
E is hidden_size or embed_dim or dims or model_dim
H is num_heads
D is head_dim
L is seq_len, in PyTorch API it's S (source len)

w_q/w_k/w_v: E x (H x D)
output/input: N x L x E
w_o: (H x D) x E
"""
class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        
        # Apply Q/K/V projections using linear (which transposes weights)
        projection_q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        
        # Apply scaled dot product attention
        attn_output = scaled_dot_product_attention_simple(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        )
        
        # Reshape back to original format and apply output projection
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_size)
        return linear(attn_output, self.wo)



def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
