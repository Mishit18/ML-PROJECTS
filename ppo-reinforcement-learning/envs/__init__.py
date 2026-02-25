"""
Custom Gym environments for PPO training.
"""

from gymnasium.envs.registration import register
from envs.custom_env import InventoryManagementEnv

# Register custom environment with Gymnasium
register(
    id='InventoryManagement-v0',
    entry_point='envs.custom_env:InventoryManagementEnv',
    max_episode_steps=200,
)

__all__ = ['InventoryManagementEnv']
