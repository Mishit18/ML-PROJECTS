"""
Actor Network for PPO

The actor network outputs the parameters of the action distribution.
For continuous actions: outputs mean and log_std for a Gaussian distribution
For discrete actions: outputs logits for a Categorical distribution

Design choices:
- Separate networks for mean and log_std in continuous case for flexibility
- Orthogonal initialization for improved gradient flow
- Tanh activation for bounded, smooth gradients
- Small final layer initialization to start with near-zero actions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.distributions import Normal, Categorical


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """
    Initialize layer with orthogonal initialization.
    
    Orthogonal initialization helps with gradient flow in deep networks
    by ensuring weight matrices are orthogonal, preventing vanishing/exploding gradients.
    
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


class Actor(nn.Module):
    """
    Actor network that outputs action distribution parameters.
    
    For continuous action spaces, outputs mean and log_std of Gaussian.
    For discrete action spaces, outputs logits for Categorical distribution.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = 'tanh',
        continuous: bool = True,
        log_std_init: float = 0.0,
    ):
        """
        Initialize actor network.
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_sizes: Tuple of hidden layer sizes
            activation: Activation function ('tanh', 'relu', 'elu')
            continuous: Whether action space is continuous
            log_std_init: Initial value for log standard deviation (continuous only)
        """
        super().__init__()
        
        self.continuous = continuous
        self.action_dim = action_dim
        
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
        
        if continuous:
            # For continuous actions: output mean
            # Use small initialization for final layer to start near zero
            self.mean_layer = layer_init(
                nn.Linear(prev_size, action_dim),
                std=0.01
            )
            
            # Log std is a learnable parameter (state-independent)
            # This approach is simpler and often works better than state-dependent std
            self.log_std = nn.Parameter(
                torch.ones(action_dim) * log_std_init
            )
        else:
            # For discrete actions: output logits
            self.logits_layer = layer_init(
                nn.Linear(prev_size, action_dim),
                std=0.01
            )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through actor network.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
        
        Returns:
            If continuous: (mean, log_std)
            If discrete: (logits, None)
        """
        features = self.network(obs)
        
        if self.continuous:
            mean = self.mean_layer(features)
            # Expand log_std to match batch size
            log_std = self.log_std.expand_as(mean)
            return mean, log_std
        else:
            logits = self.logits_layer(features)
            return logits, None
    
    def get_distribution(self, obs: torch.Tensor):
        """
        Get action distribution for given observations.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
        
        Returns:
            Distribution object (Normal or Categorical)
        """
        if self.continuous:
            mean, log_std = self.forward(obs)
            std = torch.exp(log_std)
            # Clamp std to prevent numerical instability
            std = torch.clamp(std, min=1e-6, max=10.0)
            return Normal(mean, std)
        else:
            logits, _ = self.forward(obs)
            return Categorical(logits=logits)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            deterministic: If True, return mean/mode instead of sampling
        
        Returns:
            action: Sampled action [batch_size, action_dim]
            log_prob: Log probability of action [batch_size]
        """
        dist = self.get_distribution(obs)
        
        if deterministic:
            if self.continuous:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1, keepdim=True)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        # For continuous actions, sum log probs across action dimensions
        if self.continuous and len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions.
        
        This is used during training to compute the policy loss.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            actions: Action tensor [batch_size, action_dim]
        
        Returns:
            log_prob: Log probability of actions [batch_size]
            entropy: Entropy of distribution [batch_size]
        """
        dist = self.get_distribution(obs)
        
        log_prob = dist.log_prob(actions)
        
        # For continuous actions, sum log probs across action dimensions
        if self.continuous and len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)
        
        # Entropy encourages exploration
        entropy = dist.entropy()
        if self.continuous and len(entropy.shape) > 1:
            entropy = entropy.sum(dim=-1)
        
        return log_prob, entropy
