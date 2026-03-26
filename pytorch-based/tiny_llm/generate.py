
try:
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

    from transformers import AutoTokenizer
    
    class TokenizerWrapper:
        """MLX TokenizerWrapper的PyTorch替代实现"""
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        
        def encode(self, text, **kwargs):
            return self.tokenizer.encode(text, **kwargs)
        
        def decode(self, tokens, **kwargs):
            return self.tokenizer.decode(tokens, **kwargs)
        
        @property
        def eos_token_id(self):
            return self.tokenizer.eos_token_id
        
        @property
        def bos_token_id(self):
            return self.tokenizer.bos_token_id

from .qwen2_week1 import Qwen2ModelWeek1

try:
    from .qwen2_week2 import Qwen2ModelWeek2
except ImportError:
    Qwen2ModelWeek2 = None


def simple_generate(
    model: Qwen2ModelWeek1, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    
    def _step(y, offset: int):
        import torch
        output_logits = model(y)  
        logits = output_logits[:, -1, :]  
        next_token = torch.argmax(logits, dim=-1)  
        return next_token

    if not MLX_AVAILABLE:
        import torch
        if hasattr(tokenizer, 'tokenizer'):
            actual_tokenizer = tokenizer.tokenizer
        else:
            actual_tokenizer = tokenizer
        if isinstance(prompt, str):
            tokenized_prompt = tokenizer.encode(prompt)  
            if not isinstance(tokenized_prompt, list):
                tokenized_prompt = tokenized_prompt.tolist()
        else:
            tokenized_prompt = prompt  
        tokens = torch.tensor([tokenized_prompt], dtype=torch.long)  
        
        max_new_tokens = 50  
        generated_tokens = []
        
        with torch.no_grad():
            next_token = _step(tokens, 0)
            generated_tokens.append(next_token.item())

            for i in range(1, max_new_tokens):

                new_token_tensor = torch.tensor([[generated_tokens[-1]]], dtype=torch.long)
                tokens = torch.cat([tokens, new_token_tensor], dim=1)
                offset = len(tokenized_prompt) + i - 1
                next_token = _step(tokens, offset)
                next_token_id = next_token.item()
                if hasattr(tokenizer, 'eos_token_id') and next_token_id == tokenizer.eos_token_id:
                    break
                    
                generated_tokens.append(next_token_id)
            generated_text = tokenizer.decode(generated_tokens)
            
        return generated_text
    else:
        pass


def simple_generate_with_kv_cache(
    model, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    if not MLX_AVAILABLE or Qwen2ModelWeek2 is None:
        if hasattr(model, 'generate'):
            return simple_generate(model, tokenizer, prompt)
        else:
            return "KV cache generation not available in PyTorch mode"
    else:
        pass
