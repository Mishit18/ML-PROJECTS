"""
Evaluation script for trained PPO agents.

This script evaluates a trained agent and reports:
- Mean episode return
- Standard deviation of returns
- Episode lengths
- Optional video recording

Usage:
    python training/evaluate.py --model-path results/model_final.pt --env InventoryManagement-v0 --episodes 10
"""

import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
import envs  # Register custom environments
from models.actor_critic import ActorCritic


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained PPO agent')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--env', type=str, default='InventoryManagement-v0',
                        help='Environment name')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use')
    
    return parser.parse_args()


def evaluate_agent(
    model: ActorCritic,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    device: torch.device = torch.device('cpu'),
) -> dict:
    """
    Evaluate agent performance.
    
    Args:
        model: Trained actor-critic model
        env: Gym environment
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic policy
        render: Whether to render environment
        device: Device to run model on
    
    Returns:
        Dictionary with evaluation statistics
    """
    model.eval()
    
    episode_returns = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        episode_length = 0
        
        while not done:
            # Get action from policy
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, _, _ = model.get_action_and_value(obs_tensor, deterministic=deterministic)
            
            action_np = action.cpu().numpy().flatten()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{n_episodes}: Return = {episode_return:.2f}, Length = {episode_length}")
    
    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create environment
    if args.render:
        env = gym.make(args.env, render_mode='human')
    else:
        env = gym.make(args.env)
    
    # Set seed
    env.reset(seed=args.seed)
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    
    from gymnasium.spaces import Box, Discrete
    if isinstance(env.action_space, Box):
        continuous = True
        action_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, Discrete):
        continuous = False
        action_dim = env.action_space.n
    else:
        raise ValueError(f"Unsupported action space: {type(env.action_space)}")
    
    # Initialize model
    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=(64, 64),
        activation='tanh',
        continuous=continuous,
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    print(f"\nEvaluating {args.env} for {args.episodes} episodes...")
    print("="*80)
    
    stats = evaluate_agent(
        model=model,
        env=env,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render,
        device=device,
    )
    
    print("\n" + "="*80)
    print(f"Mean Return: {stats['mean_return']:.2f} ± {stats['std_return']:.2f}")
    print(f"Min/Max: {stats['min_return']:.2f} / {stats['max_return']:.2f}")
    print(f"Mean Length: {stats['mean_length']:.1f}")
    print("="*80)
    
    env.close()


if __name__ == '__main__':
    main()
