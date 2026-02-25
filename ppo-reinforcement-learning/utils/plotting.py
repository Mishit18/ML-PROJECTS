"""
Plotting utilities for visualizing training results.

This module provides functions for creating publication-quality plots:
- Learning curves (episode returns over time)
- Multi-seed comparisons with confidence intervals
- Loss curves for training diagnostics
- Value function analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from scipy.ndimage import uniform_filter1d

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


def smooth_curve(data: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Smooth a curve using uniform filter (moving average).
    
    Args:
        data: Data to smooth
        window: Window size for smoothing
    
    Returns:
        Smoothed data
    """
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode='nearest')


def plot_learning_curves(
    returns: List[float],
    save_path: str,
    title: str = "Learning Curve",
    xlabel: str = "Episode",
    ylabel: str = "Return",
    smooth_window: int = 10,
):
    """
    Plot learning curve with smoothing.
    
    Args:
        returns: List of episode returns
        save_path: Path to save plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        smooth_window: Window size for smoothing
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = np.arange(len(returns))
    
    # Plot raw data (transparent)
    ax.plot(episodes, returns, alpha=0.3, color='blue', label='Raw')
    
    # Plot smoothed data
    if len(returns) >= smooth_window:
        smoothed = smooth_curve(np.array(returns), window=smooth_window)
        ax.plot(episodes, smoothed, color='blue', linewidth=2, label=f'Smoothed (window={smooth_window})')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_multi_seed_results(
    all_returns: Dict[int, List[float]],
    save_path: str,
    title: str = "Multi-Seed Learning Curves",
    xlabel: str = "Episode",
    ylabel: str = "Return",
    smooth_window: int = 10,
):
    """
    Plot learning curves from multiple seeds with mean and confidence interval.
    
    Args:
        all_returns: Dictionary mapping seed to list of returns
        save_path: Path to save plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        smooth_window: Window size for smoothing
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Find minimum length across all seeds
    min_length = min(len(returns) for returns in all_returns.values())
    
    # Truncate all to same length and smooth
    smoothed_returns = []
    for seed, returns in all_returns.items():
        returns_array = np.array(returns[:min_length])
        if len(returns_array) >= smooth_window:
            smoothed = smooth_curve(returns_array, window=smooth_window)
        else:
            smoothed = returns_array
        smoothed_returns.append(smoothed)
        
        # Plot individual seed (transparent)
        episodes = np.arange(len(smoothed))
        ax.plot(episodes, smoothed, alpha=0.2, color='blue')
    
    # Compute mean and std across seeds
    smoothed_returns = np.array(smoothed_returns)
    mean_returns = np.mean(smoothed_returns, axis=0)
    std_returns = np.std(smoothed_returns, axis=0)
    
    episodes = np.arange(len(mean_returns))
    
    # Plot mean
    ax.plot(episodes, mean_returns, color='blue', linewidth=2, label='Mean')
    
    # Plot confidence interval (mean ± std)
    ax.fill_between(
        episodes,
        mean_returns - std_returns,
        mean_returns + std_returns,
        alpha=0.3,
        color='blue',
        label='± 1 std'
    )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({len(all_returns)} seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_metrics(
    metrics: Dict[str, List[Tuple[int, float]]],
    save_path: str,
    title: str = "Training Metrics",
):
    """
    Plot multiple training metrics on separate subplots.
    
    Args:
        metrics: Dictionary mapping metric name to list of (step, value) tuples
        save_path: Path to save plot
        title: Plot title
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        steps, vals = zip(*values)
        ax.plot(steps, vals, linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_value_function_analysis(
    states: np.ndarray,
    true_returns: np.ndarray,
    predicted_values: np.ndarray,
    save_path: str,
):
    """
    Plot value function predictions vs true returns.
    
    This helps diagnose value function learning.
    
    Args:
        states: State observations
        true_returns: True discounted returns
        predicted_values: Value function predictions
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot: predicted vs true
    axes[0].scatter(true_returns, predicted_values, alpha=0.5)
    axes[0].plot([true_returns.min(), true_returns.max()],
                 [true_returns.min(), true_returns.max()],
                 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('True Returns')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Value Function Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = predicted_values - true_returns
    axes[1].scatter(true_returns, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('True Returns')
    axes[1].set_ylabel('Residuals (Predicted - True)')
    axes[1].set_title('Value Function Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(
    policy_losses: List[float],
    value_losses: List[float],
    entropy_losses: List[float],
    save_path: str,
):
    """
    Plot PPO loss components over training.
    
    Args:
        policy_losses: List of policy losses
        value_losses: List of value losses
        entropy_losses: List of entropy values
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    steps = np.arange(len(policy_losses))
    
    # Policy loss
    axes[0].plot(steps, policy_losses, linewidth=2, color='blue')
    axes[0].set_xlabel('Update')
    axes[0].set_ylabel('Policy Loss')
    axes[0].set_title('PPO Policy Loss (Clipped Surrogate Objective)')
    axes[0].grid(True, alpha=0.3)
    
    # Value loss
    axes[1].plot(steps, value_losses, linewidth=2, color='green')
    axes[1].set_xlabel('Update')
    axes[1].set_ylabel('Value Loss')
    axes[1].set_title('Value Function Loss (MSE)')
    axes[1].grid(True, alpha=0.3)
    
    # Entropy
    axes[2].plot(steps, entropy_losses, linewidth=2, color='orange')
    axes[2].set_xlabel('Update')
    axes[2].set_ylabel('Entropy')
    axes[2].set_title('Policy Entropy (Exploration)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
