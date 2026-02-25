"""
Advantage Estimation Utilities

This module provides functions for computing advantages, which are central
to policy gradient methods.

The advantage function A(s,a) = Q(s,a) - V(s) measures how much better
an action is compared to the average action in that state.

We implement Generalized Advantage Estimation (GAE), which provides
a bias-variance tradeoff through the lambda parameter:
- lambda=0: temporal difference residual (high bias, low variance)
- lambda=1: Monte Carlo estimation (low bias, high variance)
- lambda=0.95: empirically good tradeoff
"""

import numpy as np
import torch
from typing import Tuple


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    last_done: bool,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE computes advantages as an exponentially-weighted average of TD residuals:
    
    A_t^GAE = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
    
    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    
    This provides a smooth interpolation between:
    - TD(0): A_t = delta_t (lambda=0)
    - Monte Carlo: A_t = sum_{l=0}^{inf} gamma^l * r_{t+l} - V(s_t) (lambda=1)
    
    Args:
        rewards: Array of rewards [T]
        values: Array of value estimates [T]
        dones: Array of done flags [T]
        last_value: Value estimate for state after last step
        last_done: Whether last step was terminal
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: Array of advantage estimates [T]
        returns: Array of return estimates [T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae_lam = 0
    
    # Compute advantages backwards through time
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - last_done
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        
        # TD residual
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        
        # GAE
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
    
    # Returns are advantages + values
    returns = advantages + values
    
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize advantages to have mean 0 and standard deviation 1.
    
    This is a common technique in policy gradient methods that:
    1. Reduces variance in gradient estimates
    2. Makes learning more stable across different reward scales
    3. Acts as a form of adaptive learning rate
    
    Args:
        advantages: Advantage tensor [batch_size]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized advantages [batch_size]
    """
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def compute_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    last_done: bool,
    gamma: float = 0.99,
) -> np.ndarray:
    """
    Compute discounted returns (Monte Carlo estimates).
    
    R_t = sum_{k=0}^{inf} gamma^k * r_{t+k}
    
    This is used as the target for value function training.
    
    Args:
        rewards: Array of rewards [T]
        dones: Array of done flags [T]
        last_value: Value estimate for state after last step
        last_done: Whether last step was terminal
        gamma: Discount factor
    
    Returns:
        returns: Array of discounted returns [T]
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    
    # Start with last value (bootstrap if not terminal)
    next_return = last_value * (1.0 - last_done)
    
    # Compute returns backwards
    for t in reversed(range(T)):
        returns[t] = rewards[t] + gamma * next_return * (1.0 - dones[t])
        next_return = returns[t]
    
    return returns
