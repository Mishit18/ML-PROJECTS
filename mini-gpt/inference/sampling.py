"""
Sampling strategies for text generation.

Implements various decoding strategies:
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Beam search (optional)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy decoding: select token with highest probability.
    
    Args:
        logits: Logits of shape (B, vocab_size)
    
    Returns:
        Selected token IDs of shape (B, 1)
    """
    return torch.argmax(logits, dim=-1, keepdim=True)


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample from distribution with temperature scaling.
    
    Temperature effects:
    - temperature < 1: More confident (sharper distribution)
    - temperature = 1: Original distribution
    - temperature > 1: More random (flatter distribution)
    
    Args:
        logits: Logits of shape (B, vocab_size)
        temperature: Temperature parameter
    
    Returns:
        Sampled token IDs of shape (B, 1)
    """
    if temperature == 0:
        return greedy_decode(logits)
    
    logits = logits / temperature
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


def top_k_sampling(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Top-k sampling: sample from top k most likely tokens.
    
    Args:
        logits: Logits of shape (B, vocab_size)
        k: Number of top tokens to consider
        temperature: Temperature parameter
    
    Returns:
        Sampled token IDs of shape (B, 1)
    """
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    
    top_k_logits = top_k_logits / temperature
    
    probs = F.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    
    next_token = torch.gather(top_k_indices, -1, sampled_idx)
    
    return next_token


def top_p_sampling(
    logits: torch.Tensor,
    p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Top-p (nucleus) sampling: sample from smallest set of tokens with cumulative probability >= p.
    
    This is more dynamic than top-k, as the number of tokens varies based on the distribution.
    
    Args:
        logits: Logits of shape (B, vocab_size)
        p: Cumulative probability threshold
        temperature: Temperature parameter
    
    Returns:
        Sampled token IDs of shape (B, 1)
    """
    logits = logits / temperature
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    probs = F.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    
    next_token = torch.gather(sorted_indices, -1, sampled_idx)
    
    return next_token


def sample_token(
    logits: torch.Tensor,
    strategy: str = 'temperature',
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Sample next token using specified strategy.
    
    Args:
        logits: Logits of shape (B, vocab_size)
        strategy: Sampling strategy ('greedy', 'temperature', 'top_k', 'top_p')
        temperature: Temperature parameter
        top_k: Top-k parameter
        top_p: Top-p parameter
    
    Returns:
        Sampled token IDs of shape (B, 1)
    """
    if strategy == 'greedy':
        return greedy_decode(logits)
    
    elif strategy == 'temperature':
        return sample_with_temperature(logits, temperature)
    
    elif strategy == 'top_k':
        if top_k is None:
            raise ValueError("top_k must be specified for top_k sampling")
        return top_k_sampling(logits, top_k, temperature)
    
    elif strategy == 'top_p':
        if top_p is None:
            raise ValueError("top_p must be specified for top_p sampling")
        return top_p_sampling(logits, top_p, temperature)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


class SamplingConfig:
    """Configuration for sampling strategies."""
    
    def __init__(
        self,
        strategy: str = 'temperature',
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        self.strategy = strategy
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample using configured strategy."""
        return sample_token(
            logits,
            strategy=self.strategy,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
