"""
Reverse diffusion process for sampling.

Implements p_θ(x_{t-1} | x_t) using the trained noise prediction model.
"""

import torch
import torch.nn.functional as F


class ReverseDiffusion:
    """
    Reverse diffusion process for generating samples.
    
    Uses the trained model ε_θ(x_t, t) to iteratively denoise from pure noise.
    Supports both DDPM (stochastic) and DDIM (deterministic) sampling.
    
    Args:
        model: Trained UNet model
        diffusion_params: Dictionary of precomputed parameters
        device: Device to run on
    """
    
    def __init__(self, model, diffusion_params, device="cuda"):
        self.model = model
        self.device = device
        
        # Move parameters to device
        self.params = {}
        for key, value in diffusion_params.items():
            self.params[key] = value.to(device)
        
        self.timesteps = len(self.params["betas"])
    
    @torch.no_grad()
    def p_sample(self, x_t, t, class_labels=None, guidance_scale=1.0, clip_denoised=True):
        """
        Sample x_{t-1} from p_θ(x_{t-1} | x_t) using DDPM sampling.
        
        Args:
            x_t: (batch, channels, height, width) noisy images at timestep t
            t: (batch,) timestep indices
            class_labels: (batch,) class labels for conditioning
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
            
        Returns:
            x_{t-1}: (batch, channels, height, width) denoised images
            pred_x0: Predicted x_0 (for visualization)
        """
        batch_size = x_t.shape[0]
        
        # Predict noise with classifier-free guidance
        noise_pred = self._predict_noise_with_guidance(
            x_t, t, class_labels, guidance_scale
        )
        
        # Predict x_0 from noise
        pred_x0 = self._predict_x0_from_noise(x_t, t, noise_pred)
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute posterior mean and variance
        model_mean = self._predict_mean_from_x0(x_t, t, pred_x0)
        
        # Add noise (except at t=0)
        if t[0] > 0:
            posterior_variance = self._extract(self.params["posterior_variance"], t, x_t.shape)
            noise = torch.randn_like(x_t)
            x_prev = model_mean + torch.sqrt(posterior_variance) * noise
        else:
            x_prev = model_mean
        
        return x_prev, pred_x0
    
    @torch.no_grad()
    def p_sample_loop(self, shape, class_labels=None, guidance_scale=1.0, progress=True):
        """
        Generate samples using DDPM sampling (full reverse process).
        
        Args:
            shape: (batch, channels, height, width) shape of samples to generate
            class_labels: (batch,) class labels for conditioning
            guidance_scale: Classifier-free guidance scale
            progress: Whether to show progress bar
            
        Returns:
            samples: (batch, channels, height, width) generated images
        """
        device = self.device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Iteratively denoise
        timesteps = list(range(self.timesteps))[::-1]
        
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="DDPM Sampling")
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, t_batch, class_labels, guidance_scale)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self, x_t, t, t_prev, class_labels=None, guidance_scale=1.0, eta=0.0, clip_denoised=True):
        """
        Sample x_{t_prev} from x_t using DDIM sampling.
        
        DDIM allows deterministic sampling and can skip timesteps for faster generation.
        
        Args:
            x_t: (batch, channels, height, width) noisy images at timestep t
            t: (batch,) current timestep indices
            t_prev: (batch,) previous timestep indices
            class_labels: (batch,) class labels
            guidance_scale: Classifier-free guidance scale
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
            clip_denoised: Whether to clip predicted x_0
            
        Returns:
            x_{t_prev}: Denoised images at timestep t_prev
        """
        # Predict noise with guidance
        noise_pred = self._predict_noise_with_guidance(
            x_t, t, class_labels, guidance_scale
        )
        
        # Predict x_0
        pred_x0 = self._predict_x0_from_noise(x_t, t, noise_pred)
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Extract alpha values
        alpha_t = self._extract(self.params["alphas_cumprod"], t, x_t.shape)
        alpha_prev = self._extract(self.params["alphas_cumprod"], t_prev, x_t.shape)
        
        # Compute sigma for stochasticity
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        
        # Compute predicted noise direction
        pred_dir = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred
        
        # Compute x_{t_prev}
        x_prev = torch.sqrt(alpha_prev) * pred_x0 + pred_dir
        
        # Add stochastic noise
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise
        
        return x_prev
    
    @torch.no_grad()
    def ddim_sample_loop(self, shape, num_steps=50, class_labels=None, guidance_scale=1.0, eta=0.0, progress=True):
        """
        Generate samples using DDIM sampling with fewer steps.
        
        Args:
            shape: (batch, channels, height, width) shape of samples
            num_steps: Number of sampling steps (can be < timesteps)
            class_labels: (batch,) class labels
            guidance_scale: Classifier-free guidance scale
            eta: Stochasticity parameter
            progress: Whether to show progress bar
            
        Returns:
            samples: (batch, channels, height, width) generated images
        """
        device = self.device
        batch_size = shape[0]
        
        # Create subsequence of timesteps
        step_size = self.timesteps // num_steps
        timesteps = list(range(0, self.timesteps, step_size))[::-1]
        timesteps_prev = [0] + timesteps[:-1]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        if progress:
            from tqdm import tqdm
            iterator = tqdm(zip(timesteps, timesteps_prev), total=len(timesteps), desc="DDIM Sampling")
        else:
            iterator = zip(timesteps, timesteps_prev)
        
        for t, t_prev in iterator:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
            x = self.ddim_sample(x, t_batch, t_prev_batch, class_labels, guidance_scale, eta)
        
        return x
    
    def _predict_noise_with_guidance(self, x_t, t, class_labels, guidance_scale):
        """
        Predict noise with classifier-free guidance using batched inference.
        
        Applies guidance formula: ε̃ = ε_u + w * (ε_c - ε_u)
        
        Args:
            x_t: Noisy images
            t: Timesteps
            class_labels: Class labels (None for unconditional)
            guidance_scale: Guidance strength
            
        Returns:
            Guided noise prediction
        """
        if class_labels is None or guidance_scale == 1.0:
            # Unconditional or no guidance
            return self.model(x_t, t, class_labels, force_drop_class=True)
        
        # Compute conditional and unconditional predictions
        batch_size = x_t.shape[0]
        
        # Double the batch: [conditional, unconditional]
        x_t_doubled = torch.cat([x_t, x_t], dim=0)
        t_doubled = torch.cat([t, t], dim=0)
        c_doubled = torch.cat([class_labels, class_labels], dim=0)
        
        # Compute both predictions
        noise_cond = self.model(x_t, t, class_labels, force_drop_class=False)
        noise_uncond = self.model(x_t, t, class_labels, force_drop_class=True)
        
        # Apply classifier-free guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        return noise_pred
    
    def _predict_x0_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and noise."""
        sqrt_recip_alphas_cumprod_t = self._extract(
            self.params["sqrt_recip_alphas_cumprod"], t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.params["sqrt_recipm1_alphas_cumprod"], t, x_t.shape
        )
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def _predict_mean_from_x0(self, x_t, t, pred_x0):
        """Compute posterior mean from predicted x_0."""
        posterior_mean_coef1 = self._extract(self.params["posterior_mean_coef1"], t, x_t.shape)
        posterior_mean_coef2 = self._extract(self.params["posterior_mean_coef2"], t, x_t.shape)
        
        return posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x_t
    
    def _extract(self, a, t, x_shape):
        """Extract coefficients at timesteps and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
