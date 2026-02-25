"""
Rollout Buffer for PPO

The rollout buffer stores on-policy trajectories collected during interaction
with the environment. Unlike off-policy algorithms (e.g., DQN, SAC), PPO requires
fresh data for each update, so the buffer is cleared after each training iteration.

The buffer stores:
- Observations
- Actions
- Rewards
- Episode termination flags
- Value estimates from the critic
- Log probabilities for importance sampling

After collection, the buffer computes:
- Returns (discounted cumulative rewards)
- Advantages using Generalized Advantage Estimation (GAE)

Design choices:
- Fixed-size buffer (n_steps per environment)
- Efficient numpy arrays for storage
- Lazy computation of advantages (computed only when needed)
- Support for multiple parallel environments (currently using single environment)
"""

import numpy as np
import torch
from typing import Generator, Tuple, Optional


class RolloutBuffer:
    """
    Buffer for storing on-policy rollout data for PPO.
    
    This buffer stores trajectories and computes advantages using GAE.
    """
    
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Number of steps to store
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Storage arrays
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        
        # Computed after collection
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """
        Add a transition to the buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            done: Whether episode terminated
            value: Value estimate V(s)
            log_prob: Log probability of action
        """
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        GAE balances bias and variance in advantage estimation:
        - lambda=0: high bias, low variance (TD residual)
        - lambda=1: low bias, high variance (Monte Carlo)
        - lambda=0.95: good empirical tradeoff
        
        The advantage function A(s,a) = Q(s,a) - V(s) tells us how much better
        an action is compared to the average action in that state.
        
        Args:
            last_value: Value estimate for the last observation
            last_done: Whether the last step was terminal
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        # Initialize
        last_gae_lam = 0
        
        # Compute advantages backwards through time
        # This approach is more numerically stable than forward computation
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                # For the last step, use the provided last_value
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                # For other steps, use the stored value from next step
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            # TD residual: r + gamma * V(s') - V(s)
            # This is the one-step advantage estimate
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            
            # GAE: exponentially weighted average of TD residuals
            # A^GAE = delta_t + (gamma * lambda) * delta_{t+1} + ...
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # Returns are advantages + values
        # This is because A(s,a) = Q(s,a) - V(s), so Q(s,a) = A(s,a) + V(s)
        # And we want to fit V to match Q (the actual returns)
        self.returns = self.advantages + self.values
    
    def get(self, batch_size: Optional[int] = None) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate random mini-batches from the buffer.
        
        Args:
            batch_size: Size of mini-batches. If None, return full buffer.
        
        Yields:
            Tuple of (observations, actions, log_probs, advantages, returns, values)
        """
        indices = np.arange(self.buffer_size)
        
        if batch_size is None:
            batch_size = self.buffer_size
        
        # Shuffle indices for random mini-batches
        np.random.shuffle(indices)
        
        # Generate mini-batches
        for start_idx in range(0, self.buffer_size, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            # Convert to tensors and move to device
            yield (
                torch.as_tensor(self.observations[batch_indices], device=self.device),
                torch.as_tensor(self.actions[batch_indices], device=self.device),
                torch.as_tensor(self.log_probs[batch_indices], device=self.device),
                torch.as_tensor(self.advantages[batch_indices], device=self.device),
                torch.as_tensor(self.returns[batch_indices], device=self.device),
                torch.as_tensor(self.values[batch_indices], device=self.device),
            )
    
    def reset(self):
        """
        Reset buffer to empty state.
        
        Called after each training iteration since PPO is on-policy.
        """
        self.pos = 0
        self.full = False
