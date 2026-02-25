"""
Loss functions for language modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_language_modeling_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute cross-entropy loss for language modeling.
    
    For causal language modeling:
    - Input: tokens[:-1]
    - Target: tokens[1:]
    
    This shifting is typically done before calling this function.
    
    Args:
        logits: Model predictions of shape (B, T, vocab_size)
        labels: Target token IDs of shape (B, T)
        ignore_index: Token ID to ignore in loss computation (e.g., padding)
        reduction: Loss reduction method ('mean', 'sum', 'none')
    
    Returns:
        Loss value
    """
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    
    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    
    return loss


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss.
    
    Perplexity = exp(loss)
    
    Lower perplexity = better model
    
    Args:
        loss: Cross-entropy loss (float value)
    
    Returns:
        Perplexity value
    """
    import math
    return math.exp(loss)


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for language modeling.
    
    Instead of hard targets (one-hot), use soft targets:
    - True class: 1 - smoothing
    - Other classes: smoothing / (vocab_size - 1)
    
    This can improve generalization and calibration.
    """
    
    def __init__(
        self,
        vocab_size: int,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            smoothing: Label smoothing factor (0 = no smoothing)
            ignore_index: Token ID to ignore
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions of shape (B, T, vocab_size)
            labels: Target token IDs of shape (B, T)
        
        Returns:
            Loss value
        """
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), self.confidence)
            
            mask = (labels != self.ignore_index).unsqueeze(1)
            true_dist = true_dist * mask
        
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        
        num_tokens = mask.sum()
        loss = loss.sum() / num_tokens if num_tokens > 0 else loss.sum()
        
        return loss
