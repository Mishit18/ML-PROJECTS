"""
Critic Network for PPO

The critic network estimates the value function V(s), which represents
the expected return from a given state under the current policy.

Design choices:
- Similar architecture to actor for consistency
- Outputs single scalar value per state
- Orthogonal initialization for stable learning
- Can share features with actor (not implemented here for modularity)

The value function is used for:
1. Computing advantages (measuring how much better an action is than expected)
2. Reducing variance in policy gradient estimates
3. Bootstrapping in temporal difference learning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """
    Initialize layer with orthogonal initialization.
    
    Args:
        layer: Neural network layer
        std: Standard deviation for initialization
        bias_const: Constant value for bias initialization
    
    Returns:
        Initialized layer
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Critic(nn.Module):
    """
    Critic network that estimates state value V(s).
    
    The value function represents expected cumulative discounted reward
    from a given state following the current policy.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = 'tanh',
    ):
        """
        Initialize critic network.
        
        Args:
            obs_dim: Observation space dimension
            hidden_sizes: Tuple of hidden layer sizes
            activation: Activation function ('tanh', 'relu', 'elu')
        """
        super().__init__()
        
        # Select activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.append(layer_init(nn.Linear(prev_size, hidden_size)))
            layers.append(self.activation)
            prev_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        
        # Value head: outputs single scalar value
        # Initialize with std=1.0 (standard initialization for value functions)
        self.value_head = layer_init(
            nn.Linear(prev_size, 1),
            std=1.0
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
        
        Returns:
            value: State value estimates [batch_size, 1]
        """
        features = self.network(obs)
        value = self.value_head(features)
        return value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for given observations.
        
        This is a convenience method that's more explicit than forward().
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
        
        Returns:
            value: State value estimates [batch_size, 1]
        """
        return self.forward(obs)
