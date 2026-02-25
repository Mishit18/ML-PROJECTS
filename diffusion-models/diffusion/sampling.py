"""
Standalone sampling script for generating images from trained models.
"""

import os
import sys
import argparse
import yaml
import torch
from torchvision.utils import save_image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from diffusion.schedules import get_beta_schedule, compute_diffusion_params
from diffusion.reverse import ReverseDiffusion
from utils.checkpoints import load_model_for_inference


def main():
    """Generate samples from a trained diffusion model."""
    parser = argparse.ArgumentParser(description='Generate samples from trained diffusion model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='./generated_samples', help='Output directory')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='Classifier-free guidance scale')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM steps (0 for full DDPM)')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta parameter')
    parser.add_argument('--class_label', type=int, default=None, help='Class label for conditional generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config from checkpoint directory
    checkpoint_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {config_path}")
    
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    
    all_samples = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in range(num_batches):
        batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        
        shape = (
            batch_size,
            config['model']['in_channels'],
            config['model']['image_size'],
            config['model']['image_size'],
        )
        
        # Prepare class labels
        if config['conditioning']['enabled']:
            if args.class_label is not None:
                class_labels = torch.full((batch_size,), args.class_label, device=device, dtype=torch.long)
            else:
                # Generate samples for all classes
                class_labels = torch.arange(batch_size, device=device) % config['conditioning']['num_classes']
        else:
            class_labels = None
        
        # Sample
        if args.ddim_steps > 0:
            samples = reverse_diffusion.ddim_sample_loop(
                shape,
                num_steps=args.ddim_steps,
                class_labels=class_labels,
                guidance_scale=args.guidance_scale,
                eta=args.ddim_eta,
                progress=True,
            )
        else:
            samples = reverse_diffusion.p_sample_loop(
                shape,
                class_labels=class_labels,
                guidance_scale=args.guidance_scale,
                progress=True,
            )
        
        all_samples.append(samples.cpu())
        
        print(f"Generated batch {i+1}/{num_batches}")
    
    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]
    
    # Save as grid
    grid_path = os.path.join(args.output_dir, 'samples_grid.png')
    save_image(all_samples, grid_path, nrow=8, normalize=True, value_range=(-1, 1))
    print(f"Saved sample grid to {grid_path}")
    
    # Save individual images
    for i, sample in enumerate(all_samples):
        sample_path = os.path.join(args.output_dir, f'sample_{i:04d}.png')
        save_image(sample, sample_path, normalize=True, value_range=(-1, 1))
    
    print(f"Saved {len(all_samples)} individual samples to {args.output_dir}")


if __name__ == '__main__':
    main()
