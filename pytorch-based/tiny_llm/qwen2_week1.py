import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import RoPE
from .layer_norm import RMSNorm
from .mlp import Qwen2MLP
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class Qwen2ModelWeek1(nn.Module):
    def __init__(self, model_input):
        super().__init__()
        
        if isinstance(model_input, str):
            self._init_from_path(model_input)
        elif hasattr(model_input, 'config'):
            self._init_from_hf_model(model_input)
        else:
            raise ValueError("Input must be either model path/name or HuggingFace model instance")
    
    def _init_from_hf_model(self, hf_model):
        self.config = hf_model.config
        
        self.hf_model = hf_model
        
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.num_heads = getattr(self.config, 'num_attention_heads', 12)
        self.num_layers = getattr(self.config, 'num_hidden_layers', 12)
        self.tie_word_embeddings = getattr(self.config, 'tie_word_embeddings', True)
    
    def _init_model_architecture(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.tie_word_embeddings = getattr(config, 'tie_word_embeddings', True)
        
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([
            self._build_transformer_layer()
            for _ in range(self.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.hidden_size)
        
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)
    
    def _build_transformer_layer(self):
        return nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        if hasattr(self, 'hf_model'):
            with torch.no_grad():
                outputs = self.hf_model(input_ids, **kwargs)
                if hasattr(outputs, 'logits'):
                    return outputs.logits
                else:
                    return outputs
        else:
            x = self.embedding(input_ids)
            
            attention_mask = self._prepare_attention_mask(input_ids)
            
            for layer in self.layers:
                x = layer(x, src_mask=attention_mask)
            
            x = self.final_norm(x)
            
            if self.tie_word_embeddings:
                logits = F.linear(x, self.embedding.weight)
            else:
                logits = self.lm_head(x)
            
            return logits
    
    def _prepare_attention_mask(self, input_ids):
        seq_len = input_ids.size(-1)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1)
        return mask.masked_fill(mask, float('-inf'))
    
    def generate(self, *args, **kwargs):
        if hasattr(self, 'hf_model'):
            return self.hf_model.generate(*args, **kwargs)
        else:
            raise NotImplementedError("Generate method not available without HuggingFace model")
    
    @property
    def device(self):
        if hasattr(self, 'hf_model'):
            return next(self.hf_model.parameters()).device
        else:
            return next(self.parameters()).device

        
class Qwen2TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_query_heads, intermediate_size, mlx_model):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_query_heads = num_query_heads
        self.head_dim = hidden_size // num_heads
        self.query_head_dim = hidden_size // num_query_heads
        
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        
        self.wq = nn.Linear(hidden_size, num_query_heads * self.query_head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_query_heads * self.query_head_dim, hidden_size, bias=False)
        
        I = intermediate_size or hidden_size * 4
        self.mlp = Qwen2MLP(
            hidden_size, I,
            mlx_model.mlp_gate,
            mlx_model.mlp_up,
            mlx_model.mlp_down
        )
        
        self.rope = RoPE(self.query_head_dim, seq_len=2048, traditional=False)
    
    def forward(self, x: torch.Tensor, offset: int = 0, mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        
        B, L, E = x.shape
        q = self.wq(x).view(B, L, self.num_query_heads, self.query_head_dim)
        k = self.wk(x).view(B, L, self.num_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim)
        
        q = self.rope(q, offset=slice(offset, offset + L))
        k = self.rope(k, offset=slice(offset, offset + L))
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_scores = torch.matmul(q.to(torch.float32), k.transpose(-1, -2).to(torch.float32))
        attn_scores = attn_scores / (self.query_head_dim ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores + mask
        
        attn_probs = F.softmax(attn_scores, dim=-1).to(v.dtype)
        attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        x = self.wo(attn_output)
        x = x + residual
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        
        return x