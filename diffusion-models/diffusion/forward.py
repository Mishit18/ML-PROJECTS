"""
Forward diffusion process implementation.

Implements q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1 - ᾱ_t) * I)
"""

import torch
import torch.nn.functional as F


class ForwardDiffusion:
    """
    Forward diffusion process that gradually adds Gaussian noise to data.
    
    Implements the closed-form solution:
        x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε, where ε ~ N(0, I)
    
    This allows sampling x_t at any timestep t without iterating through all steps.
    
    Args:
        diffusion_params: Dictionary of precomputed parameters from schedules.py
        device: Device to place tensors on
    """
    
    def __init__(self, diffusion_params, device="cuda"):
        self.device = device
        
        # Move all parameters to device
        self.params = {}
        for key, value in diffusion_params.items():
            self.params[key] = value.to(device)
        
        self.timesteps = len(self.params["betas"])
    
    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) using the closed-form solution.
        
        Args:
            x_0: (batch, channels, height, width) clean images
            t: (batch,) timestep indices in [0, T-1]
            noise: (batch, channels, height, width) noise tensor (optional, sampled if None)
            
        Returns:
            x_t: (batch, channels, height, width) noisy images at timestep t
            noise: The noise that was added (useful for training)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Extract coefficients for the given timesteps
        sqrt_alphas_cumprod_t = self._extract(self.params["sqrt_alphas_cumprod"], t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.params["sqrt_one_minus_alphas_cumprod"], t, x_0.shape
        )
        
        # Apply forward diffusion: x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the posterior q(x_{t-1} | x_t, x_0).
        
        This is used in the reverse process when we know x_0.
        
        Args:
            x_0: (batch, channels, height, width) clean images
            x_t: (batch, channels, height, width) noisy images at timestep t
            t: (batch,) timestep indices
            
        Returns:
            posterior_mean: Mean of q(x_{t-1} | x_t, x_0)
            posterior_variance: Variance of q(x_{t-1} | x_t, x_0)
            posterior_log_variance: Log variance (for numerical stability)
        """
        # Extract coefficients
        posterior_mean_coef1 = self._extract(self.params["posterior_mean_coef1"], t, x_0.shape)
        posterior_mean_coef2 = self._extract(self.params["posterior_mean_coef2"], t, x_0.shape)
        
        # Compute posterior mean
        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        
        # Extract variance
        posterior_variance = self._extract(self.params["posterior_variance"], t, x_0.shape)
        posterior_log_variance = self._extract(self.params["posterior_log_variance"], t, x_0.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def predict_x0_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise.
        
        Rearranging x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε:
            x_0 = (x_t - √(1 - ᾱ_t) * ε) / √ᾱ_t
        
        Args:
            x_t: (batch, channels, height, width) noisy images
            t: (batch,) timestep indices
            noise: (batch, channels, height, width) predicted noise
            
        Returns:
            x_0: (batch, channels, height, width) predicted clean images
        """
        sqrt_recip_alphas_cumprod_t = self._extract(
            self.params["sqrt_recip_alphas_cumprod"], t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.params["sqrt_recipm1_alphas_cumprod"], t, x_t.shape
        )
        
        # x_0 = (x_t - √(1 - ᾱ_t) * ε) / √ᾱ_t
        x_0 = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
        
        return x_0
    
    def _extract(self, a, t, x_shape):
        """
        Extract coefficients at specified timesteps and reshape for broadcasting.
        
        Args:
            a: (timesteps,) tensor of coefficients
            t: (batch,) timestep indices
            x_shape: Shape of the data tensor (for broadcasting)
            
        Returns:
            (batch, 1, 1, 1) tensor of coefficients
        """
        batch_size = t.shape[0]
        out = a.gather(0, t)
        
        # Reshape to (batch, 1, 1, 1) for broadcasting with (batch, C, H, W)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def visualize_forward_process(self, x_0, timesteps_to_show=None, save_path=None):
        """
        Visualize the forward diffusion process on a batch of images.
        
        Args:
            x_0: (batch, channels, height, width) clean images
            timesteps_to_show: List of timesteps to visualize (default: evenly spaced)
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if timesteps_to_show is None:
            # Show 10 evenly spaced timesteps
            timesteps_to_show = np.linspace(0, self.timesteps - 1, 10, dtype=int)
        
        batch_size = min(8, x_0.shape[0])
        num_timesteps = len(timesteps_to_show)
        
        fig, axes = plt.subplots(batch_size, num_timesteps, figsize=(num_timesteps * 2, batch_size * 2))
        
        if batch_size == 1:
            axes = axes[np.newaxis, :]
        
        for i in range(batch_size):
            for j, t in enumerate(timesteps_to_show):
                # Sample noisy image at timestep t
                t_tensor = torch.tensor([t], device=x_0.device)
                x_t, _ = self.q_sample(x_0[i:i+1], t_tensor)
                
                # Convert to numpy and denormalize
                img = x_t[0].cpu().permute(1, 2, 0).numpy()
                img = (img + 1) / 2  # Assuming images are in [-1, 1]
                img = np.clip(img, 0, 1)
                
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                
                if i == 0:
                    axes[i, j].set_title(f't={t}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Forward process visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
