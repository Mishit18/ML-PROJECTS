"""
Logger for tracking training metrics.

This module provides a simple logging interface for tracking:
- Episode returns and lengths
- Training losses and metrics
- Other custom metrics

The logger can output to:
- Console (stdout)
- CSV files for data analysis
- TensorBoard for visualization (optional)
"""

import os
import csv
import time
from typing import Dict, Any, Optional, List
from collections import defaultdict
import numpy as np


class Logger:
    """
    Logger for tracking and saving training metrics.
    """
    
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = False,
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Storage for metrics
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.episode_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False
        
        # CSV file
        self.csv_path = os.path.join(log_dir, 'metrics.csv')
        self.csv_file = None
        self.csv_writer = None
    
    def log_scalar(self, key: str, value: float, step: int):
        """
        Log a scalar metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Training step/timestep
        """
        self.metrics[key].append((step, value))
        
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar(key, value, step)
    
    def log_episode(self, episode_return: float, episode_length: int, step: int):
        """
        Log episode statistics.
        
        Args:
            episode_return: Total episode return
            episode_length: Episode length in steps
            step: Training step/timestep
        """
        self.episode_metrics['return'].append(episode_return)
        self.episode_metrics['length'].append(episode_length)
        
        self.log_scalar('episode/return', episode_return, step)
        self.log_scalar('episode/length', episode_length, step)
    
    def log_training_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log training metrics (losses, etc.).
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step/timestep
        """
        for key, value in metrics.items():
            self.log_scalar(f'train/{key}', value, step)
    
    def print_progress(
        self,
        step: int,
        total_steps: int,
        fps: Optional[float] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Print training progress to console.
        
        Args:
            step: Current training step
            total_steps: Total training steps
            fps: Frames per second (optional)
            extra_metrics: Additional metrics to print (optional)
        """
        elapsed_time = time.time() - self.start_time
        
        # Compute recent episode statistics
        if len(self.episode_metrics['return']) > 0:
            recent_returns = self.episode_metrics['return'][-100:]
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            
            recent_lengths = self.episode_metrics['length'][-100:]
            mean_length = np.mean(recent_lengths)
        else:
            mean_return = 0.0
            std_return = 0.0
            mean_length = 0.0
        
        # Build progress string
        progress_str = (
            f"Step {step}/{total_steps} ({100*step/total_steps:.1f}%) | "
            f"Time: {elapsed_time:.0f}s | "
            f"Return: {mean_return:.2f} ± {std_return:.2f} | "
            f"Length: {mean_length:.1f}"
        )
        
        if fps is not None:
            progress_str += f" | FPS: {fps:.0f}"
        
        if extra_metrics:
            for key, value in extra_metrics.items():
                if isinstance(value, float):
                    progress_str += f" | {key}: {value:.4f}"
                else:
                    progress_str += f" | {key}: {value}"
        
        print(progress_str)
    
    def save_metrics_csv(self, verbose: bool = False):
        """
        Save all metrics to CSV file.
        
        Args:
            verbose: Whether to print confirmation
        """
        if not self.metrics:
            return
        
        # Collect all unique steps
        all_steps = set()
        for values in self.metrics.values():
            for step, _ in values:
                all_steps.add(step)
        
        all_steps = sorted(all_steps)
        
        # Create CSV with all metrics
        with open(self.csv_path, 'w', newline='') as f:
            # Get all metric names
            metric_names = sorted(self.metrics.keys())
            writer = csv.writer(f)
            writer.writerow(['step'] + metric_names)
            
            # Write data for each step
            for step in all_steps:
                row = [step]
                for metric_name in metric_names:
                    # Find value for this step
                    value = None
                    for s, v in self.metrics[metric_name]:
                        if s == step:
                            value = v
                            break
                    row.append(value if value is not None else '')
                writer.writerow(row)
        
        if verbose:
            print(f"Metrics saved to {self.csv_path}")
    
    def get_episode_returns(self) -> List[float]:
        """
        Get all episode returns.
        
        Returns:
            List of episode returns
        """
        return self.episode_metrics['return']
    
    def get_episode_lengths(self) -> List[int]:
        """
        Get all episode lengths.
        
        Returns:
            List of episode lengths
        """
        return self.episode_metrics['length']
    
    def close(self):
        """
        Close logger and save final data.
        """
        self.save_metrics_csv()
        
        if self.writer is not None:
            self.writer.close()
