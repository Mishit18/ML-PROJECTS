"""
Generate sample grids for visual comparison.
"""

import os
import sys
import argparse
import yaml
import torch
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from diffusion.schedules import get_beta_schedule, compute_diffusion_params
from diffusion.reverse import ReverseDiffusion
from utils.checkpoints import load_model_for_inference


def generate_class_conditional_grid(model, reverse_diffusion, config, device, guidance_scales=[1.0, 2.0, 3.0, 5.0]):
    """
    Generate grid showing effect of different guidance scales.
    
    Args:
        model: Trained model
        reverse_diffusion: ReverseDiffusion instance
        config: Config dict
        device: Device
        guidance_scales: List of guidance scales to compare
        
    Returns:
        grid: Image grid tensor
    """
    num_classes = config['conditioning']['num_classes']
    samples_per_class = 8
    
    all_samples = []
    
    for guidance_scale in guidance_scales:
        print(f"Generating samples with guidance_scale={guidance_scale}")
        
        # Generate samples for each class
        class_samples = []
        for class_idx in range(num_classes):
            shape = (
                samples_per_class,
                config['model']['in_channels'],
                config['model']['image_size'],
                config['model']['image_size'],
            )
            
            class_labels = torch.full((samples_per_class,), class_idx, device=device, dtype=torch.long)
            
            samples = reverse_diffusion.p_sample_loop(
                shape,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                progress=False,
            )
            
            class_samples.append(samples)
        
        all_samples.append(torch.cat(class_samples, dim=0))
    
    # Stack all samples
    all_samples = torch.cat(all_samples, dim=0)
    
    return all_samples


def main():
    """Generate sample grids for visual comparison."""
    parser = argparse.ArgumentParser(description='Generate sample grids')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='./sample_grids', help='Output directory')
    parser.add_argument('--guidance_scales', type=float, nargs='+', default=[1.0, 2.0, 3.0, 5.0],
                        help='Guidance scales to compare')
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if config['conditioning']['enabled']:
        # Generate class-conditional grid
        print("Generating class-conditional samples...")
        samples = generate_class_conditional_grid(
            model, reverse_diffusion, config, device, args.guidance_scales
        )
        
        # Save grid
        grid_path = os.path.join(args.output_dir, 'guidance_comparison.png')
        save_image(samples, grid_path, nrow=8, normalize=True, value_range=(-1, 1))
        print(f"Saved guidance comparison to {grid_path}")
    else:
        # Generate unconditional samples
        print("Generating unconditional samples...")
        shape = (
            64,
            config['model']['in_channels'],
            config['model']['image_size'],
            config['model']['image_size'],
        )
        
        samples = reverse_diffusion.p_sample_loop(shape, progress=True)
        
        grid_path = os.path.join(args.output_dir, 'unconditional_samples.png')
        save_image(samples, grid_path, nrow=8, normalize=True, value_range=(-1, 1))
        print(f"Saved samples to {grid_path}")


if __name__ == '__main__':
    main()
