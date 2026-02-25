"""
Checkpoint management utilities.
"""

import torch
import os


def save_checkpoint(path, model, optimizer, ema, epoch, global_step):
    """
    Save training checkpoint.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        ema: EMA state
        epoch: Current epoch
        global_step: Global training step
    """
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, ema=None, device='cuda'):
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        ema: EMA to load state into (optional)
        device: Device to load tensors to
        
    Returns:
        epoch: Epoch number from checkpoint
        global_step: Global step from checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    
    return epoch, global_step


def load_model_for_inference(path, model, use_ema=True, device='cuda'):
    """
    Load model for inference/sampling.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        use_ema: Whether to use EMA weights
        device: Device to load to
        
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(path, map_location=device)
    
    if use_ema and 'ema_state_dict' in checkpoint:
        # Load EMA weights
        ema_state = checkpoint['ema_state_dict']
        shadow = ema_state['shadow']
        
        # Create state dict from EMA shadow
        model_state = {}
        for name, param in model.named_parameters():
            if name in shadow:
                model_state[name] = shadow[name]
        
        model.load_state_dict(model_state, strict=False)
    else:
        # Load regular weights
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model
