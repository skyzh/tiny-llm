import torch
import torch.nn.functional as F


def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature == 0.0:
        next_token = torch.argmax(logits, dim=-1)
    else:
        scaled_logits = logits / temperature
        
        probs = F.softmax(scaled_logits, dim=-1)
        
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return next_token


def top_p_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    
    sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True, dim=-1)
    
    sorted_probs = torch.exp(sorted_log_probs)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    mask = cumsum_probs <= p
    mask[..., 0] = True
    
    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
    
    keep_mask.scatter_(-1, sorted_indices, mask)
    
    masked_logits = torch.where(keep_mask, logits, torch.full_like(logits, float('-inf')))
    
    return temperature_sampling(masked_logits, temperature)


def top_k_sampling(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    vocab_size = logits.size(-1)
    k = min(k, vocab_size)
    
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, top_k_indices, True)
    
    masked_logits = torch.where(mask, logits, torch.full_like(logits, float('-inf')))
    
    return temperature_sampling(masked_logits, temperature)


class Sampler:
    
    def __init__(self, method: str = "temperature", **kwargs):
        self.method = method
        self.params = kwargs
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.method == "temperature":
            temperature = self.params.get("temperature", 1.0)
            return temperature_sampling(logits, temperature)
        
        elif self.method == "top_p":
            p = self.params.get("p", 0.9)
            temperature = self.params.get("temperature", 1.0)
            return top_p_sampling(logits, p, temperature)
        
        elif self.method == "top_k":
            k = self.params.get("k", 50)
            temperature = self.params.get("temperature", 1.0)
            return top_k_sampling(logits, k, temperature)
        
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")


def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    return temperature_sampling(logits, temperature=0.0) 