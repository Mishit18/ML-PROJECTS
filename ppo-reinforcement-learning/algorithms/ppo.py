"""
Proximal Policy Optimization (PPO) Algorithm

This module implements PPO following the paper:
"Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

PPO is an on-policy policy gradient method that:
1. Collects trajectories using the current policy
2. Computes advantages using Generalized Advantage Estimation (GAE)
3. Updates the policy using a clipped surrogate objective
4. Updates the value function using mean squared error loss
5. Adds entropy bonus for exploration

Key innovations of PPO:
- Clipped objective prevents destructively large policy updates
- Multiple epochs of minibatch updates for sample efficiency
- Simple to implement and tune
- State-of-the-art performance on many tasks

The loss function is:
L = L_CLIP + c1 * L_VF - c2 * H

Where:
- L_CLIP: Clipped surrogate objective (policy loss)
- L_VF: Value function loss (mean squared error)
- H: Entropy bonus (encourages exploration)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
from models.actor_critic import ActorCritic
from buffers.rollout_buffer import RolloutBuffer
from utils.advantage import normalize_advantages


class PPO:
    """
    Proximal Policy Optimization algorithm.
    
    This implementation includes:
    - Clipped surrogate objective
    - GAE for advantage estimation
    - Value function loss with optional clipping
    - Entropy regularization
    - Gradient clipping
    - Learning rate annealing
    - KL divergence monitoring
    """
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        clip_value_loss: bool = False,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        hidden_sizes: tuple = (64, 64),
        activation: str = 'tanh',
        device: str = 'auto',
        seed: Optional[int] = None,
    ):
        """
        Initialize PPO algorithm.
        
        Args:
            env: Gym environment
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to collect before update
            batch_size: Mini-batch size for SGD
            n_epochs: Number of epochs to train on each batch
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            clip_value_loss: Whether to clip value loss
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: Target KL divergence for early stopping
            hidden_sizes: Hidden layer sizes for networks
            activation: Activation function
            device: Device to use ('auto', 'cpu', 'cuda')
            seed: Random seed
        """
        self.env = env
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.clip_value_loss = clip_value_loss
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if hasattr(env, 'seed'):
                env.seed(seed)
            if hasattr(env, 'action_space'):
                env.action_space.seed(seed)
        
        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        
        # Determine if action space is continuous
        from gymnasium.spaces import Box, Discrete
        if isinstance(env.action_space, Box):
            self.continuous = True
            self.action_dim = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            self.continuous = False
            self.action_dim = env.action_space.n
        else:
            raise ValueError(f"Unsupported action space: {type(env.action_space)}")
        
        # Initialize actor-critic model
        self.model = ActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            continuous=self.continuous,
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        
        # Initialize rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=n_steps,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim if self.continuous else 1,
            device=self.device,
        )
        
        # Training state
        self.num_timesteps = 0
        self.num_updates = 0
        
        # For learning rate annealing
        self.initial_lr = learning_rate
    
    def collect_rollouts(self) -> Dict[str, Any]:
        """
        Collect rollouts using current policy.
        
        This fills the rollout buffer with n_steps of experience.
        
        Returns:
            Dictionary with rollout statistics
        """
        self.model.eval()
        
        episode_returns = []
        episode_lengths = []
        current_episode_return = 0
        current_episode_length = 0
        
        obs, _ = self.env.reset()
        
        for step in range(self.n_steps):
            # Convert observation to tensor
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Get action and value from policy
            with torch.no_grad():
                action, log_prob, value = self.model.get_action_and_value(obs_tensor)
            
            # Convert action to numpy
            action_np = action.cpu().numpy().flatten()
            if not self.continuous:
                action_np = int(action_np[0])  # Discrete actions need scalar
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Store transition
            self.buffer.add(
                obs=obs,
                action=action.cpu().numpy().flatten(),
                reward=reward,
                done=done,
                value=value.cpu().item(),
                log_prob=log_prob.cpu().item(),
            )
            
            # Update episode statistics
            current_episode_return += reward
            current_episode_length += 1
            self.num_timesteps += 1
            
            # Handle episode end
            if done:
                episode_returns.append(current_episode_return)
                episode_lengths.append(current_episode_length)
                current_episode_return = 0
                current_episode_length = 0
                obs, _ = self.env.reset()
            else:
                obs = next_obs
        
        # Compute value for last observation (for bootstrapping)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.model.get_value(obs_tensor).cpu().item()
        
        # Compute advantages and returns
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=done,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        
        return {
            'episode_returns': episode_returns,
            'episode_lengths': episode_lengths,
            'mean_return': np.mean(episode_returns) if episode_returns else 0.0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0.0,
        }
    
    def train(self) -> Dict[str, float]:
        """
        Train on collected rollouts.
        
        This performs multiple epochs of minibatch SGD on the rollout data.
        
        Returns:
            Dictionary with training statistics
        """
        self.model.train()
        
        # Storage for logging
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divs = []
        clip_fractions = []
        
        # Train for n_epochs
        for epoch in range(self.n_epochs):
            # Generate random mini-batches
            for batch in self.buffer.get(self.batch_size):
                obs, actions, old_log_probs, advantages, returns, old_values = batch
                
                # Normalize advantages (improves stability)
                advantages = normalize_advantages(advantages)
                
                # Evaluate actions with current policy
                log_probs, entropy, values = self.model.evaluate_actions(obs, actions)
                values = values.squeeze(-1)
                
                # Policy loss (clipped surrogate objective)
                # ratio = pi_new / pi_old
                ratio = torch.exp(log_probs - old_log_probs)
                
                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                if self.clip_value_loss:
                    # Clip value function updates (optional, from PPO paper)
                    value_pred_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.clip_epsilon,
                        self.clip_epsilon
                    )
                    value_loss_1 = (values - returns).pow(2)
                    value_loss_2 = (value_pred_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = 0.5 * (returns - values).pow(2).mean()
                
                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients (prevents exploding gradients)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
                
                # Compute KL divergence (for monitoring)
                with torch.no_grad():
                    kl_div = (old_log_probs - log_probs).mean()
                    kl_divs.append(kl_div.item())
                    
                    # Compute clip fraction (fraction of ratios that were clipped)
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_epsilon).float())
                    clip_fractions.append(clip_fraction.item())
            
            # Early stopping based on KL divergence
            if self.target_kl is not None:
                if np.mean(kl_divs) > self.target_kl:
                    break
        
        self.num_updates += 1
        
        # Reset buffer for next rollout
        self.buffer.reset()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'kl_div': np.mean(kl_divs),
            'clip_fraction': np.mean(clip_fractions),
        }
    
    def learn(self, total_timesteps: int, log_interval: int = 2048, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the agent for a specified number of timesteps.
        
        Args:
            total_timesteps: Total number of environment steps
            log_interval: Logging frequency in timesteps
        
        Returns:
            Dictionary with training history
        """
        history = {
            'timesteps': [],
            'episode_returns': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
        }
        
        num_iterations = total_timesteps // self.n_steps
        
        for iteration in range(num_iterations):
            # Collect rollouts
            rollout_stats = self.collect_rollouts()
            
            # Train on rollouts
            train_stats = self.train()
            
            # Log
            if verbose and self.num_timesteps % log_interval < self.n_steps:
                print(f"\nTimestep: {self.num_timesteps}/{total_timesteps}")
                print(f"  Mean Return: {rollout_stats['mean_return']:.2f}")
                print(f"  Mean Length: {rollout_stats['mean_length']:.1f}")
                print(f"  Policy Loss: {train_stats['policy_loss']:.4f}")
                print(f"  Value Loss: {train_stats['value_loss']:.4f}")
                print(f"  Entropy: {train_stats['entropy']:.4f}")
                print(f"  KL Div: {train_stats['kl_div']:.4f}")
                print(f"  Clip Fraction: {train_stats['clip_fraction']:.3f}")
            
            # Store history
            history['timesteps'].append(self.num_timesteps)
            if rollout_stats['episode_returns']:
                history['episode_returns'].extend(rollout_stats['episode_returns'])
                history['episode_lengths'].extend(rollout_stats['episode_lengths'])
            history['policy_losses'].append(train_stats['policy_loss'])
            history['value_losses'].append(train_stats['value_loss'])
            history['entropies'].append(train_stats['entropy'])
        
        return history
    
    def save(self, path: str, verbose: bool = False):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            verbose: Whether to print confirmation
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
            'num_updates': self.num_updates,
        }, path)
        if verbose:
            print(f"Model saved to {path}")
    
    def load(self, path: str, verbose: bool = False):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
            verbose: Whether to print confirmation
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_timesteps = checkpoint['num_timesteps']
        self.num_updates = checkpoint['num_updates']
        if verbose:
            print(f"Model loaded from {path}")
