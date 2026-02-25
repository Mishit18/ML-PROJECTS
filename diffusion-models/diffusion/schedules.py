"""
Noise schedules for the forward diffusion process.

Implements various beta schedules that control the rate of noise addition:
- Linear schedule: Simple linear interpolation
- Cosine schedule: Smoother noise addition (better for high-resolution images)
"""

import torch
import numpy as np
import math


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Linear schedule from Ho et al. (2020).
    
    β_t increases linearly from beta_start to beta_end.
    
    Args:
        timesteps: Number of diffusion steps T
        beta_start: Starting value β_1
        beta_end: Ending value β_T
        
    Returns:
        (timesteps,) tensor of beta values
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from Nichol & Dhariwal (2021).
    
    Provides smoother noise addition, especially beneficial for high-resolution images.
    Based on: ᾱ_t = f(t) / f(0), where f(t) = cos((t/T + s)/(1 + s) * π/2)^2
    
    Args:
        timesteps: Number of diffusion steps T
        s: Small offset to prevent β_t from being too small near t=0
        
    Returns:
        (timesteps,) tensor of beta values
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    
    # Compute alpha_bar using cosine function
    alpha_bar = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]  # Normalize so alpha_bar[0] = 1
    
    # Compute beta from alpha_bar: β_t = 1 - α_t = 1 - (ᾱ_t / ᾱ_{t-1})
    alpha = alpha_bar[1:] / alpha_bar[:-1]
    beta = 1 - alpha
    
    # Clip to prevent numerical issues
    return torch.clip(beta, 0.0001, 0.9999)


def get_beta_schedule(schedule_name, timesteps, **kwargs):
    """
    Factory function to get beta schedule by name.
    
    Args:
        schedule_name: "linear" or "cosine"
        timesteps: Number of diffusion steps
        **kwargs: Additional arguments for specific schedules
        
    Returns:
        (timesteps,) tensor of beta values
    """
    if schedule_name == "linear":
        return linear_beta_schedule(
            timesteps,
            beta_start=kwargs.get("beta_start", 0.0001),
            beta_end=kwargs.get("beta_end", 0.02),
        )
    elif schedule_name == "cosine":
        return cosine_beta_schedule(timesteps, s=kwargs.get("s", 0.008))
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")


def compute_diffusion_params(betas):
    """
    Precompute all parameters needed for forward and reverse diffusion.
    
    Given β_t, compute:
    - α_t = 1 - β_t
    - ᾱ_t = ∏_{i=1}^t α_i (cumulative product)
    - √ᾱ_t, √(1 - ᾱ_t) for forward process
    - Posterior variance for reverse process
    
    Args:
        betas: (timesteps,) tensor of beta values
        
    Returns:
        Dictionary containing all precomputed parameters
    """
    timesteps = len(betas)
    
    # α_t = 1 - β_t
    alphas = 1.0 - betas
    
    # ᾱ_t = ∏_{i=1}^t α_i
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # Clamp to prevent numerical instability
    alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-5)
    
    # ᾱ_{t-1} (shifted by 1, with ᾱ_0 = 1)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    
    # Square roots for forward process: x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    # For computing x_0 from x_t and predicted noise
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
    
    # Posterior variance: β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    # Clip to prevent log(0)
    posterior_variance = torch.clamp(posterior_variance, min=1e-20)
    
    # Log variance for numerical stability
    posterior_log_variance = torch.log(posterior_variance)
    
    # Posterior mean coefficients
    # μ̃_t(x_t, x_0) = (√ᾱ_{t-1} * β_t) / (1 - ᾱ_t) * x_0 + (√α_t * (1 - ᾱ_{t-1})) / (1 - ᾱ_t) * x_t
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
    
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
        "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "posterior_log_variance": posterior_log_variance,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }


def visualize_schedule(betas, save_path=None):
    """
    Visualize the noise schedule and related parameters.
    
    Args:
        betas: (timesteps,) tensor of beta values
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt
    
    params = compute_diffusion_params(betas)
    timesteps = len(betas)
    t = np.arange(timesteps)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot beta schedule
    axes[0, 0].plot(t, betas.numpy())
    axes[0, 0].set_title("Beta Schedule")
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("β_t")
    axes[0, 0].grid(True)
    
    # Plot alpha_bar
    axes[0, 1].plot(t, params["alphas_cumprod"].numpy())
    axes[0, 1].set_title("Cumulative Alpha (ᾱ_t)")
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("ᾱ_t")
    axes[0, 1].grid(True)
    
    # Plot signal and noise coefficients
    axes[1, 0].plot(t, params["sqrt_alphas_cumprod"].numpy(), label="√ᾱ_t (signal)")
    axes[1, 0].plot(t, params["sqrt_one_minus_alphas_cumprod"].numpy(), label="√(1-ᾱ_t) (noise)")
    axes[1, 0].set_title("Forward Process Coefficients")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("Coefficient")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot posterior variance
    axes[1, 1].plot(t, params["posterior_variance"].numpy())
    axes[1, 1].set_title("Posterior Variance")
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("β̃_t")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Schedule visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
