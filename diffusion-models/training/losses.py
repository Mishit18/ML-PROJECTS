"""
Loss functions for training diffusion models.
"""

import torch
import torch.nn.functional as F


def simple_diffusion_loss(model, x_0, t, forward_diffusion, class_labels=None):
    """
    Simple MSE loss for noise prediction: L = E[||ε - ε_θ(x_t, t)||²]
    
    Args:
        model: UNet model that predicts noise
        x_0: (batch, channels, height, width) clean images
        t: (batch,) timestep indices
        forward_diffusion: ForwardDiffusion instance
        class_labels: (batch,) class labels for conditioning
        
    Returns:
        loss: Scalar MSE loss
        metrics: Dictionary of additional metrics for logging
    """
    # Sample noise
    noise = torch.randn_like(x_0)
    
    # Forward diffusion: add noise to images
    x_t, _ = forward_diffusion.q_sample(x_0, t, noise=noise)
    
    # Predict noise
    noise_pred = model(x_t, t, class_labels)
    
    # Compute MSE loss
    loss = F.mse_loss(noise_pred, noise)
    
    # Additional metrics
    metrics = {
        'loss': loss.item(),
        'noise_mse': loss.item(),
    }
    
    return loss, metrics


def vlb_loss(model, x_0, t, forward_diffusion, class_labels=None):
    """
    Variational lower bound (VLB) loss.
    
    More principled but typically not necessary for good results.
    Included for completeness and research purposes.
    
    Args:
        model: UNet model
        x_0: Clean images
        t: Timesteps
        forward_diffusion: ForwardDiffusion instance
        class_labels: Class labels
        
    Returns:
        loss: VLB loss
        metrics: Dictionary of metrics
    """
    # Sample noise
    noise = torch.randn_like(x_0)
    
    # Forward diffusion
    x_t, _ = forward_diffusion.q_sample(x_0, t, noise=noise)
    
    # Predict noise
    noise_pred = model(x_t, t, class_labels)
    
    # Predict x_0 from noise
    pred_x0 = forward_diffusion.predict_x0_from_noise(x_t, t, noise_pred)
    
    # Compute true posterior
    true_mean, true_var, true_log_var = forward_diffusion.q_posterior_mean_variance(x_0, x_t, t)
    
    # Compute model posterior
    model_mean, model_var, model_log_var = forward_diffusion.q_posterior_mean_variance(pred_x0, x_t, t)
    
    # KL divergence between Gaussians
    kl = 0.5 * (
        -1.0
        + model_log_var
        - true_log_var
        + true_var / model_var
        + (true_mean - model_mean) ** 2 / model_var
    )
    kl = kl.mean(dim=[1, 2, 3])
    
    # Simple loss for t=0
    decoder_nll = -torch.log(torch.tensor(1.0))  # Placeholder
    
    # Combine losses
    loss = torch.where(t == 0, decoder_nll, kl).mean()
    
    metrics = {
        'loss': loss.item(),
        'kl': kl.mean().item(),
    }
    
    return loss, metrics


def hybrid_loss(model, x_0, t, forward_diffusion, class_labels=None, vlb_weight=0.001):
    """
    Hybrid loss combining simple MSE and VLB.
    
    L = L_simple + λ * L_vlb
    
    Args:
        model: UNet model
        x_0: Clean images
        t: Timesteps
        forward_diffusion: ForwardDiffusion instance
        class_labels: Class labels
        vlb_weight: Weight for VLB term
        
    Returns:
        loss: Combined loss
        metrics: Dictionary of metrics
    """
    simple_loss, simple_metrics = simple_diffusion_loss(model, x_0, t, forward_diffusion, class_labels)
    vlb, vlb_metrics = vlb_loss(model, x_0, t, forward_diffusion, class_labels)
    
    loss = simple_loss + vlb_weight * vlb
    
    metrics = {
        'loss': loss.item(),
        'simple_loss': simple_metrics['loss'],
        'vlb_loss': vlb_metrics['loss'],
    }
    
    return loss, metrics
