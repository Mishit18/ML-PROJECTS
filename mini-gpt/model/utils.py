"""
Utility functions for model operations.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count number of parameters in model.
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def initialize_weights(model: nn.Module, init_std: float = 0.02):
    """
    Initialize model weights using GPT-2 style initialization.
    
    Args:
        model: PyTorch model
        init_std: Standard deviation for normal initialization
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'ln' in name or 'layernorm' in name:
                # LayerNorm weights initialized to 1
                nn.init.ones_(param)
            else:
                # Linear layer weights
                nn.init.normal_(param, mean=0.0, std=init_std)
        elif 'bias' in name:
            # All biases initialized to 0
            nn.init.zeros_(param)


def get_lr_scheduler_lambda(
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    Create learning rate schedule function.
    
    Schedule:
    - Linear warmup from 0 to 1 over warmup_steps
    - Cosine decay from 1 to min_lr_ratio over remaining steps
    
    Args:
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of max lr
    
    Returns:
        Lambda function for lr_scheduler
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return lr_lambda


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity = exp(loss)
    
    Args:
        loss: Cross-entropy loss value
    
    Returns:
        Perplexity value
    """
    import math
    return math.exp(loss)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get device for training/inference.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
    
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    step: int,
    loss: float,
    path: str,
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        step: Current training step
        loss: Current loss value
        path: Path to save checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'step': step,
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> dict:
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
    
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', float('inf')),
    }


def estimate_mfu(model: nn.Module, batch_size: int, seq_len: int, dt: float) -> float:
    """
    Estimate Model FLOPs Utilization (MFU).
    
    This is a rough estimate of hardware efficiency.
    
    Args:
        model: The model
        batch_size: Batch size
        seq_len: Sequence length
        dt: Time taken for forward+backward pass (seconds)
    
    Returns:
        MFU as a percentage
    """
    # Rough FLOP estimate for transformer forward+backward
    # Forward: ~6 * N * B * T * d^2 (where N = params, d = d_model)
    # Backward: ~2x forward
    N = count_parameters(model)
    flops_per_iter = 6 * N * batch_size * seq_len
    flops_per_iter *= 3  # Account for backward pass
    
    # Hardware peak FLOPS (example: A100 = 312 TFLOPS for FP16)
    # This is a placeholder - adjust based on actual hardware
    peak_flops = 312e12  # A100 GPU
    
    # Achieved FLOPS
    achieved_flops = flops_per_iter / dt
    
    # MFU
    mfu = achieved_flops / peak_flops * 100
    
    return mfu
