"""
Custom Inventory Management Environment

This environment simulates a realistic inventory management problem where an agent
must decide how much inventory to order each period to balance:
1. Holding costs (penalty for excess inventory)
2. Stockout costs (penalty for insufficient inventory)
3. Order costs (fixed and variable costs for placing orders)

The environment features:
- Stochastic demand following realistic patterns
- Capacity constraints
- Lead time delays
- Time-varying demand (day-of-week effects)
- Continuous action space (order quantity)

This is a non-trivial problem requiring the agent to learn:
- Demand forecasting from historical data
- Risk management (safety stock levels)
- Cost optimization under uncertainty
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class InventoryManagementEnv(gym.Env):
    """
    Inventory Management Environment with stochastic demand.
    
    State Space:
        - Current inventory level (normalized)
        - Demand history (last 7 periods)
        - Day of week (one-hot encoded)
        - Pending orders (in transit)
    
    Action Space:
        - Order quantity (continuous, [0, max_order])
    
    Reward:
        - Negative cost = -(holding_cost + stockout_cost + order_cost)
    
    Dynamics:
        - Demand is stochastic with day-of-week seasonality
        - Orders arrive after lead_time periods
        - Inventory cannot go negative (lost sales model)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(
        self,
        max_inventory: int = 100,
        max_order: int = 50,
        holding_cost: float = 1.0,
        stockout_cost: float = 10.0,
        order_fixed_cost: float = 5.0,
        order_variable_cost: float = 2.0,
        lead_time: int = 2,
        demand_mean: float = 15.0,
        demand_std: float = 5.0,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the inventory management environment.
        
        Args:
            max_inventory: Maximum inventory capacity
            max_order: Maximum order quantity per period
            holding_cost: Cost per unit of inventory held per period
            stockout_cost: Cost per unit of unmet demand
            order_fixed_cost: Fixed cost for placing any order
            order_variable_cost: Variable cost per unit ordered
            lead_time: Number of periods before order arrives
            demand_mean: Mean daily demand
            demand_std: Standard deviation of demand
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        # Environment parameters
        self.max_inventory = max_inventory
        self.max_order = max_order
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.order_fixed_cost = order_fixed_cost
        self.order_variable_cost = order_variable_cost
        self.lead_time = lead_time
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.render_mode = render_mode
        
        # Day-of-week demand multipliers (weekends have lower demand)
        self.day_multipliers = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7])
        
        # State components:
        # - Current inventory (1)
        # - Demand history (7)
        # - Day of week one-hot (7)
        # - Pending orders (lead_time)
        state_dim = 1 + 7 + 7 + self.lead_time
        
        # Observation space: continuous values normalized to reasonable ranges
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space: order quantity [0, max_order]
        self.action_space = spaces.Box(
            low=0.0,
            high=float(self.max_order),
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.inventory = 0.0
        self.demand_history = []
        self.pending_orders = []
        self.day_of_week = 0
        self.timestep = 0
        
        # For rendering
        self.episode_costs = []
        self.episode_inventory = []
        self.episode_demand = []
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Initialize inventory at a reasonable starting point
        # (approximately 1-2 days of expected demand)
        self.inventory = self.demand_mean * 1.5
        
        # Initialize demand history with mean demand
        self.demand_history = [self.demand_mean] * 7
        
        # No pending orders initially
        self.pending_orders = [0.0] * self.lead_time
        
        # Start on a random day of week
        self.day_of_week = self.np_random.integers(0, 7)
        
        self.timestep = 0
        
        # Reset rendering data
        self.episode_costs = []
        self.episode_inventory = []
        self.episode_demand = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment.
        
        Args:
            action: Order quantity (continuous)
        
        Returns:
            observation: Next state
            reward: Reward (negative cost)
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Extract and clip action to valid range
        order_quantity = np.clip(action[0], 0.0, self.max_order)
        
        # Generate stochastic demand with day-of-week seasonality
        day_multiplier = self.day_multipliers[self.day_of_week]
        demand = self.np_random.normal(
            self.demand_mean * day_multiplier,
            self.demand_std
        )
        demand = max(0.0, demand)  # Demand cannot be negative
        
        # Process incoming order (from lead_time periods ago)
        incoming_order = self.pending_orders.pop(0)
        self.inventory += incoming_order
        
        # Ensure inventory doesn't exceed capacity
        self.inventory = min(self.inventory, self.max_inventory)
        
        # Meet demand (lost sales if insufficient inventory)
        sales = min(demand, self.inventory)
        stockout = demand - sales
        self.inventory -= sales
        
        # Calculate costs
        holding_cost = self.holding_cost * self.inventory
        stockout_cost = self.stockout_cost * stockout
        
        # Order cost: fixed cost if ordering anything, plus variable cost
        if order_quantity > 0:
            order_cost = self.order_fixed_cost + self.order_variable_cost * order_quantity
        else:
            order_cost = 0.0
        
        total_cost = holding_cost + stockout_cost + order_cost
        
        # Reward is negative cost (we want to minimize cost)
        reward = -total_cost
        
        # Add current order to pending orders queue
        self.pending_orders.append(order_quantity)
        
        # Update demand history (rolling window)
        self.demand_history.pop(0)
        self.demand_history.append(demand)
        
        # Advance day of week
        self.day_of_week = (self.day_of_week + 1) % 7
        
        self.timestep += 1
        
        # Store for rendering
        self.episode_costs.append(total_cost)
        self.episode_inventory.append(self.inventory)
        self.episode_demand.append(demand)
        
        # Episode termination conditions
        # Natural termination: none (continuous task)
        # Truncation: handled by TimeLimit wrapper in Gymnasium
        terminated = False
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        info['cost'] = total_cost
        info['holding_cost'] = holding_cost
        info['stockout_cost'] = stockout_cost
        info['order_cost'] = order_cost
        info['demand'] = demand
        info['sales'] = sales
        info['stockout'] = stockout
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector from current state.
        
        Returns:
            observation: State vector
        """
        # Normalize inventory to [0, 1] range
        inventory_normalized = self.inventory / self.max_inventory
        
        # Normalize demand history
        demand_history_normalized = np.array(self.demand_history) / (self.demand_mean * 2)
        
        # One-hot encode day of week
        day_onehot = np.zeros(7)
        day_onehot[self.day_of_week] = 1.0
        
        # Normalize pending orders
        pending_orders_normalized = np.array(self.pending_orders) / self.max_order
        
        # Concatenate all components
        observation = np.concatenate([
            [inventory_normalized],
            demand_history_normalized,
            day_onehot,
            pending_orders_normalized
        ]).astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about current state.
        
        Returns:
            info: Dictionary with state information
        """
        return {
            'inventory': self.inventory,
            'timestep': self.timestep,
            'day_of_week': self.day_of_week,
        }
    
    def render(self):
        """
        Render the environment for visualization.
        """
        if self.render_mode == 'human':
            print(f"Step {self.timestep}: Inventory={self.inventory:.1f}, "
                  f"Day={self.day_of_week}, "
                  f"Avg Cost={np.mean(self.episode_costs[-10:]) if self.episode_costs else 0:.2f}")
        
        # RGB array rendering could be implemented with matplotlib
        # but is not included in this implementation
        return None
    
    def close(self):
        """Clean up resources."""
        pass
