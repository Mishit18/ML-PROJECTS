"""
Actor-Critic Combined Model

This module combines the actor and critic into a single model for convenience.
While they could share a feature extractor, we keep them separate for:
1. Modularity and clarity
2. Flexibility for different learning rates if needed
3. Easier debugging and analysis

The combined model provides a unified interface for:
- Getting actions and values simultaneously during rollout collection
- Evaluating actions and values during training updates
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from models.actor import Actor
from models.critic import Critic


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic model for PPO.
    
    This wraps separate actor and critic networks and provides
    convenient methods for common operations.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = 'tanh',
        continuous: bool = True,
    ):
        """
        Initialize actor-critic model.
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_sizes: Tuple of hidden layer sizes
            activation: Activation function
            continuous: Whether action space is continuous
        """
        super().__init__()
        
        self.continuous = continuous
        
        # Initialize actor and critic with same architecture
        self.actor = Actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            continuous=continuous,
        )
        
        self.critic = Critic(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
        
        Returns:
            action_params: Action distribution parameters
            value: State value estimates [batch_size, 1]
        """
        action_params = self.actor(obs)
        value = self.critic(obs)
        return action_params, value
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value for given observation.
        
        This is used during rollout collection.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            deterministic: If True, return deterministic action
        
        Returns:
            action: Sampled action [batch_size, action_dim]
            log_prob: Log probability of action [batch_size]
            value: State value estimate [batch_size, 1]
        """
        action, log_prob = self.actor.get_action(obs, deterministic)
        value = self.critic.get_value(obs)
        return action, log_prob, value
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions: get log probs, entropy, and values.
        
        This is used during training to compute losses.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            actions: Action tensor [batch_size, action_dim]
        
        Returns:
            log_prob: Log probability of actions [batch_size]
            entropy: Entropy of action distribution [batch_size]
            value: State value estimates [batch_size, 1]
        """
        log_prob, entropy = self.actor.evaluate_actions(obs, actions)
        value = self.critic.get_value(obs)
        return log_prob, entropy, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for given observation.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
        
        Returns:
            value: State value estimate [batch_size, 1]
        """
        return self.critic.get_value(obs)
