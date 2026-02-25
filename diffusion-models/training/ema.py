"""
Exponential Moving Average (EMA) for model weights.

EMA provides more stable and higher-quality samples by maintaining
a moving average of model parameters during training.
"""

import torch
import copy


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of model weights that are updated as:
        θ_ema = decay * θ_ema + (1 - decay) * θ_model
    
    Args:
        model: PyTorch model to track
        decay: Decay rate (typically 0.9999)
        device: Device to store EMA parameters
    """
    
    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device if device is not None else torch.device('cpu')
        
        # Create shadow parameters
        self.shadow = {}
        self.original = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
    
    def update(self, model):
        """
        Update EMA parameters with current model parameters.
        
        Args:
            model: Current model with updated parameters
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data.to(self.device)
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model):
        """
        Replace model parameters with EMA parameters.
        
        Stores original parameters for later restoration.
        
        Args:
            model: Model to apply EMA parameters to
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device)
    
    def restore(self, model):
        """
        Restore original model parameters.
        
        Args:
            model: Model to restore parameters to
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.original
                param.data = self.original[name]
        self.original = {}
    
    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
