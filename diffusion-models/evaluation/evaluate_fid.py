"""
Fréchet Inception Distance (FID) evaluation.

FID measures the quality of generated images by comparing their statistics
in Inception feature space to real images.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from scipy import linalg
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from diffusion.schedules import get_beta_schedule, compute_diffusion_params
from diffusion.reverse import ReverseDiffusion
from utils.checkpoints import load_model_for_inference
from data.dataset_utils import get_dataset


class InceptionFeatureExtractor(torch.nn.Module):
    """
    Extract features from Inception-v3 for FID computation.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        self.device = device
        
        # Remove final classification layer
        self.inception.fc = torch.nn.Identity()
    
    @torch.no_grad()
    def forward(self, x):
        """
        Extract features from images.
        
        Args:
            x: (batch, 3, H, W) images in [-1, 1]
            
        Returns:
            (batch, 2048) feature vectors
        """
        # Resize to 299x299 for Inception
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1] then to Inception's expected range
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        
        # Extract features
        features = self.inception(x)
        
        return features


def calculate_activation_statistics(images, feature_extractor, batch_size=50):
    """
    Calculate mean and covariance of Inception features.
    
    Args:
        images: (N, 3, H, W) tensor of images
        feature_extractor: InceptionFeatureExtractor instance
        batch_size: Batch size for feature extraction
        
    Returns:
        mu: Mean of features
        sigma: Covariance of features
    """
    feature_extractor.eval()
    
    num_images = len(images)
    num_batches = (num_images + batch_size - 1) // batch_size
    
    features_list = []
    
    for i in tqdm(range(num_batches), desc="Extracting features"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_images)
        batch = images[start_idx:end_idx].to(feature_extractor.device)
        
        features = feature_extractor(batch)
        features_list.append(features.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    
    # Calculate statistics
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet Inception Distance.
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1: Mean of real features
        sigma1: Covariance of real features
        mu2: Mean of generated features
        sigma2: Covariance of generated features
        eps: Small constant for numerical stability
        
    Returns:
        fid: FID score (lower is better)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    # Calculate squared difference of means
    diff = mu1 - mu2
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle numerical errors
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Handle imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    # Calculate FID
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    return fid


def generate_samples(model, reverse_diffusion, config, num_samples, batch_size, device, guidance_scale=1.0):
    """
    Generate samples from the model.
    
    Args:
        model: Trained model
        reverse_diffusion: ReverseDiffusion instance
        config: Configuration dict
        num_samples: Number of samples to generate
        batch_size: Batch size
        device: Device
        guidance_scale: Guidance scale
        
    Returns:
        samples: (num_samples, 3, H, W) tensor of generated images
    """
    model.eval()
    
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Generating samples"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        shape = (
            current_batch_size,
            config['model']['in_channels'],
            config['model']['image_size'],
            config['model']['image_size'],
        )
        
        # Generate class labels if conditional
        if config['conditioning']['enabled']:
            class_labels = torch.arange(current_batch_size, device=device) % config['conditioning']['num_classes']
        else:
            class_labels = None
        
        # Generate samples
        samples = reverse_diffusion.p_sample_loop(
            shape,
            class_labels=class_labels,
            guidance_scale=guidance_scale,
            progress=False,
        )
        
        all_samples.append(samples.cpu())
    
    return torch.cat(all_samples, dim=0)[:num_samples]


def main():
    """Calculate FID score for a trained model."""
    parser = argparse.ArgumentParser(description='Calculate FID score')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples for FID')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    checkpoint_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = UNet(
        image_size=config['model']['image_size'],
        in_channels=config['model']['in_channels'],
        model_channels=config['model']['model_channels'],
        out_channels=config['model']['out_channels'],
        num_res_blocks=config['model']['num_res_blocks'],
        attention_resolutions=config['model']['attention_resolutions'],
        channel_mult=config['model']['channel_mult'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        use_scale_shift_norm=config['model']['use_scale_shift_norm'],
        num_classes=config['conditioning']['num_classes'] if config['conditioning']['enabled'] else None,
        class_dropout_prob=config['conditioning'].get('dropout_prob', 0.1),
    ).to(device)
    
    # Load checkpoint
    model = load_model_for_inference(args.checkpoint, model, use_ema=True, device=device)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create diffusion process
    betas = get_beta_schedule(
        config['diffusion']['beta_schedule'],
        config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
    )
    diffusion_params = compute_diffusion_params(betas)
    reverse_diffusion = ReverseDiffusion(model, diffusion_params, device)
    
    # Load real images
    print("Loading real images...")
    _, test_dataset = get_dataset(config)
    
    # Sample real images
    real_images = []
    for i in range(min(args.num_samples, len(test_dataset))):
        img, _ = test_dataset[i]
        real_images.append(img)
    real_images = torch.stack(real_images)
    print(f"Loaded {len(real_images)} real images")
    
    # Generate fake images
    print(f"Generating {args.num_samples} samples...")
    fake_images = generate_samples(
        model, reverse_diffusion, config,
        args.num_samples, args.batch_size, device,
        guidance_scale=args.guidance_scale
    )
    print(f"Generated {len(fake_images)} samples")
    
    # Create feature extractor
    print("Creating feature extractor...")
    feature_extractor = InceptionFeatureExtractor(device)
    
    # Calculate statistics for real images
    print("Calculating statistics for real images...")
    mu_real, sigma_real = calculate_activation_statistics(real_images, feature_extractor, args.batch_size)
    
    # Calculate statistics for fake images
    print("Calculating statistics for generated images...")
    mu_fake, sigma_fake = calculate_activation_statistics(fake_images, feature_extractor, args.batch_size)
    
    # Calculate FID
    fid_score = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    
    print(f"\n{'='*50}")
    print(f"FID Score: {fid_score:.2f}")
    print(f"{'='*50}\n")
    
    # Save results
    results_path = os.path.join(checkpoint_dir, 'fid_results.txt')
    with open(results_path, 'a') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Guidance Scale: {args.guidance_scale}\n")
        f.write(f"FID Score: {fid_score:.2f}\n")
        f.write(f"{'-'*50}\n")
    
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
