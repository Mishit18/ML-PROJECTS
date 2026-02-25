"""
Optimizer configuration for training.

Implements AdamW with weight decay, following best practices from GPT-2/GPT-3.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def configure_optimizers(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    device_type: str = 'cuda',
) -> torch.optim.Optimizer:
    """
    Configure AdamW optimizer with weight decay.
    
    Key insights:
    - Separate parameters into decay and no-decay groups
    - No weight decay on biases and LayerNorm parameters
    - Weight decay only on 2D parameters (weight matrices)
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        eps: Adam epsilon for numerical stability
        device_type: Device type ('cuda' or 'cpu')
    
    Returns:
        Configured optimizer
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if param.ndim < 2 or 'ln' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    use_fused = device_type == 'cuda' and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        fused=use_fused if use_fused else False,
    )
    
    return optimizer
