import os
import numpy as np
from typing import Union, Optional, Tuple, Any

# Determine backend from environment variable
BACKEND = os.getenv("BACKEND", "mlx")

if BACKEND == "mlx":
    import mlx.core as mx
    import mlx.nn as nn
    
    # Backend-specific types
    BackendTensor = mx.array
    BackendStream = mx.Stream
    BackendDtype = mx.Dtype
    
    # Available streams and precisions for MLX
    AVAILABLE_STREAMS = [mx.cpu]
    AVAILABLE_STREAMS_IDS = ["cpu"]
    PRECISIONS = [mx.float32, mx.float16]
    PRECISION_IDS = ["f32", "f16"]
    
    # Backend functions
    def be_random_uniform(shape: Tuple[int, ...], dtype: BackendDtype) -> BackendTensor:
        return mx.random.uniform(shape=shape, dtype=dtype)
    
    def be_softmax(x: BackendTensor, axis: int) -> BackendTensor:
        return mx.softmax(x, axis=axis)
    
    def be_scaled_dot_product_attention(
        q: BackendTensor, 
        k: BackendTensor, 
        v: BackendTensor, 
        scale: float = 1.0,
        mask: Optional[BackendTensor] = None
    ) -> BackendTensor:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    
    def be_addmm(bias: BackendTensor, input: BackendTensor, weight: BackendTensor) -> BackendTensor:
        return mx.addmm(bias, input, weight.T)
    
    def be_stream(stream: BackendStream):
        return mx.stream(stream)
    
    def be_reshape(tensor: BackendTensor, shape: Tuple[int, ...]) -> BackendTensor:
        return tensor.reshape(shape)
    
    def be_tensor(array: np.ndarray, device: str = "cpu") -> BackendTensor:
        return mx.array(array)
    
    def be_device() -> str:
        return "cpu"
    
    def be_dtype_to_np(dtype: BackendDtype) -> np.dtype:
        if dtype == mx.float32:
            return np.float32
        elif dtype == mx.float16:
            return np.float16
        else:
            raise ValueError(f"Unsupported MLX dtype: {dtype}")
    
    def be_np_to_dtype(np_dtype: np.dtype) -> BackendDtype:
        if np_dtype == np.float32:
            return mx.float32
        elif np_dtype == np.float16:
            return mx.float16
        else:
            raise ValueError(f"Unsupported numpy dtype: {np_dtype}")
    
    # MultiHeadAttention for MLX
    class BackendMultiHeadAttention:
        def __init__(self, embed_dim: int, num_heads: int):
            self.mha = nn.MultiHeadAttention(embed_dim, num_heads)
        
        def __call__(self, query: BackendTensor, key: BackendTensor, value: BackendTensor, mask: Optional[BackendTensor] = None) -> BackendTensor:
            return self.mha(query, key, value, mask=mask)
        
        def set_weights(self, q_weight: BackendTensor, k_weight: BackendTensor, v_weight: BackendTensor, out_weight: BackendTensor):
            self.mha.query_proj.weight = q_weight
            self.mha.key_proj.weight = k_weight
            self.mha.value_proj.weight = v_weight
            self.mha.out_proj.weight = out_weight

else:  # PyTorch backend
    import torch
    import torch.nn as nn
    
    # Backend-specific types
    BackendTensor = torch.Tensor
    BackendStream = str  # For PyTorch, we'll use string representation
    BackendDtype = torch.dtype
    
    # Available streams and precisions for PyTorch
    AVAILABLE_STREAMS = ["cpu"]
    if torch.cuda.is_available():
        AVAILABLE_STREAMS.append("cuda")
    AVAILABLE_STREAMS_IDS = AVAILABLE_STREAMS
    PRECISIONS = [np.float32, np.float16]
    PRECISION_IDS = ["f32", "f16"]
    
    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Backend functions
    def be_random_uniform(shape: Tuple[int, ...], dtype: np.dtype) -> BackendTensor:
        return torch.rand(shape, dtype=torch.float32 if dtype == np.float32 else torch.float16, device=TORCH_DEVICE)
    
    def be_softmax(x: BackendTensor, axis: int) -> BackendTensor:
        return torch.nn.functional.softmax(x, dim=axis)
    
    def be_scaled_dot_product_attention(
        q: BackendTensor, 
        k: BackendTensor, 
        v: BackendTensor, 
        scale: float = 1.0,
        mask: Optional[BackendTensor] = None
    ) -> BackendTensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    
    def be_addmm(bias: BackendTensor, input: BackendTensor, weight: BackendTensor) -> BackendTensor:
        return torch.nn.functional.linear(input, weight, bias)
    
    def be_stream(stream: str):
        # PyTorch doesn't have explicit stream context like MLX
        # We'll use a dummy context manager
        class DummyContext:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return DummyContext()
    
    def be_reshape(tensor: BackendTensor, shape: Tuple[int, ...]) -> BackendTensor:
        return tensor.reshape(shape)
    
    def be_tensor(array: np.ndarray, device: str = "cpu") -> BackendTensor:
        return torch.tensor(array, device=TORCH_DEVICE)
    
    def be_device() -> str:
        return str(TORCH_DEVICE)
    
    def be_dtype_to_np(dtype: np.dtype) -> np.dtype:
        return dtype
    
    def be_np_to_dtype(np_dtype: np.dtype) -> np.dtype:
        return np_dtype
    
    # MultiHeadAttention for PyTorch
    class BackendMultiHeadAttention:
        def __init__(self, embed_dim: int, num_heads: int):
            self.embed_dim = embed_dim
            self.num_heads = num_heads
        
        def __call__(self, query: BackendTensor, key: BackendTensor, value: BackendTensor, mask: Optional[BackendTensor] = None) -> BackendTensor:
            # Transpose for PyTorch's expected format
            query_t = query.transpose(0, 1)
            key_t = key.transpose(0, 1)
            value_t = value.transpose(0, 1)
            
            output, _ = torch.nn.functional.multi_head_attention_forward(
                query_t, key_t, value_t,
                num_heads=self.num_heads,
                q_proj_weight=self.q_weight,
                k_proj_weight=self.k_weight,
                v_proj_weight=self.v_weight,
                out_proj_weight=self.out_weight,
                embed_dim_to_check=self.embed_dim,
                in_proj_weight=None,
                in_proj_bias=None,
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0.0,
                out_proj_bias=None,
                use_separate_proj_weight=True,
                attn_mask=mask,
            )
            return output.transpose(0, 1)
        
        def set_weights(self, q_weight: BackendTensor, k_weight: BackendTensor, v_weight: BackendTensor, out_weight: BackendTensor):
            self.q_weight = q_weight
            self.k_weight = k_weight
            self.v_weight = v_weight
            self.out_weight = out_weight
