"""
Training script for PPO.

This script trains a PPO agent on a specified environment and saves:
- Model checkpoints at regular intervals
- Training logs and metrics
- Learning curves and visualizations
- Performance statistics

Usage:
    python training/train.py --env CartPole-v1 --total-timesteps 100000 --seed 42
    python training/train.py --env InventoryManagement-v0 --total-timesteps 200000 --seed 42
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
import torch
import gymnasium as gym

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
import envs  # Register custom environments
from algorithms.ppo import PPO
from utils.logger import Logger
from utils.plotting import plot_learning_curves, plot_loss_curves


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PPO agent')
    
    # Environment
    parser.add_argument('--env', type=str, default='InventoryManagement-v0',
                        help='Environment name')
    
    # Training
    parser.add_argument('--total-timesteps', type=int, default=200000,
                        help='Total number of timesteps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='Number of steps per rollout')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Mini-batch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                        help='PPO clip epsilon')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='Max gradient norm')
    
    # Network architecture
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[64, 64],
                        help='Hidden layer sizes')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'relu', 'elu'],
                        help='Activation function')
    
    # Logging and saving
    parser.add_argument('--log-interval', type=int, default=2048,
                        help='Logging interval (timesteps)')
    parser.add_argument('--save-interval', type=int, default=50000,
                        help='Model save interval (timesteps)')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override args with config values
        for key, value in config.get('ppo', {}).items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    return args


def main():
    """Main training function."""
    args = parse_args()
    
    # Create experiment name
    if args.exp_name is None:
        args.exp_name = f"{args.env}_{args.seed}_{int(time.time())}"
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Starting training: {args.env}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Seed: {args.seed}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*80}\n")
    
    # Save configuration
    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Create environment
    env = gym.make(args.env)
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if hasattr(env, 'reset'):
        env.reset(seed=args.seed)
    
    # Initialize logger
    logger = Logger(log_dir=save_dir, use_tensorboard=True)
    
    # Initialize PPO agent
    agent = PPO(
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
        device='auto',
        seed=args.seed,
    )
    
    # Training loop
    start_time = time.time()
    num_iterations = args.total_timesteps // args.n_steps
    
    all_episode_returns = []
    all_episode_lengths = []
    
    for iteration in range(num_iterations):
        # Collect rollouts
        rollout_stats = agent.collect_rollouts()
        
        # Train on rollouts
        train_stats = agent.train()
        
        # Log episode statistics
        if rollout_stats['episode_returns']:
            for ep_return, ep_length in zip(rollout_stats['episode_returns'], 
                                            rollout_stats['episode_lengths']):
                logger.log_episode(ep_return, ep_length, agent.num_timesteps)
                all_episode_returns.append(ep_return)
                all_episode_lengths.append(ep_length)
        
        # Log training statistics
        logger.log_training_metrics(train_stats, agent.num_timesteps)
        
        # Print progress
        if agent.num_timesteps % args.log_interval < args.n_steps:
            elapsed_time = time.time() - start_time
            fps = agent.num_timesteps / elapsed_time
            
            logger.print_progress(
                step=agent.num_timesteps,
                total_steps=args.total_timesteps,
                fps=fps,
                extra_metrics={
                    'policy_loss': train_stats['policy_loss'],
                    'value_loss': train_stats['value_loss'],
                    'entropy': train_stats['entropy'],
                    'kl_div': train_stats['kl_div'],
                }
            )
        
        # Save model checkpoint
        if agent.num_timesteps % args.save_interval < args.n_steps:
            model_path = os.path.join(save_dir, 'models', f'model_{agent.num_timesteps}.pt')
            agent.save(model_path)
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'models', 'model_final.pt')
    agent.save(final_model_path)
    
    # Close logger
    logger.close()
    
    # Final summary
    if all_episode_returns:
        final_mean = np.mean(all_episode_returns[-100:])
        logger.print_progress(
            step=args.total_timesteps,
            total_steps=args.total_timesteps,
            extra_metrics={
                'status': 'COMPLETE',
                'final_return': f'{final_mean:.2f}',
                'total_episodes': len(all_episode_returns)
            }
        )
    
    # Generate plots
    if all_episode_returns:
        plot_path = os.path.join(save_dir, 'plots', 'learning_curve.png')
        plot_learning_curves(
            returns=all_episode_returns,
            save_path=plot_path,
            title=f'PPO Learning Curve - {args.env}',
            xlabel='Episode',
            ylabel='Return',
        )
        
        # Loss curves
        policy_losses = [m[1] for m in logger.metrics.get('train/policy_loss', [])]
        value_losses = [m[1] for m in logger.metrics.get('train/value_loss', [])]
        entropies = [m[1] for m in logger.metrics.get('train/entropy', [])]
        
        if policy_losses:
            loss_plot_path = os.path.join(save_dir, 'plots', 'loss_curves.png')
            plot_loss_curves(
                policy_losses=policy_losses,
                value_losses=value_losses,
                entropy_losses=entropies,
                save_path=loss_plot_path,
            )
    
    env.close()


if __name__ == '__main__':
    main()
