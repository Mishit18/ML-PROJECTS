"""
Main training script for diffusion models.

Supports both unconditional and conditional (classifier-free guidance) training.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from diffusion.schedules import get_beta_schedule, compute_diffusion_params
from diffusion.forward import ForwardDiffusion
from diffusion.reverse import ReverseDiffusion
from training.ema import EMA
from training.losses import simple_diffusion_loss
from data.dataset_utils import get_dataset
from utils.logger import setup_logger
from utils.checkpoints import save_checkpoint, load_checkpoint


def train_epoch(model, dataloader, optimizer, forward_diffusion, device, epoch, writer, global_step, config, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    nan_count = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device) if config['conditioning']['enabled'] else None
        
        # Sample random timesteps
        batch_size = images.shape[0]
        t = torch.randint(0, forward_diffusion.timesteps, (batch_size,), device=device)
        
        # Compute loss with mixed precision if enabled
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, metrics = simple_diffusion_loss(model, images, t, forward_diffusion, labels)
        else:
            loss, metrics = simple_diffusion_loss(model, images, t, forward_diffusion, labels)
        
        # Check for numerical instability
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            print(f"\nWarning: NaN/Inf loss detected at step {global_step}, skipping batch ({nan_count} total)")
            
            if nan_count > 10:
                raise RuntimeError(f"Training unstable: {nan_count} NaN/Inf losses detected. Stopping training.")
            
            continue
        
        # Backward pass
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config['training']['gradient_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('train/loss', loss.item(), global_step)
            for key, value in metrics.items():
                if key != 'loss':
                    writer.add_scalar(f'train/{key}', value, global_step)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        global_step += 1
    
    avg_loss = total_loss / max(num_batches - nan_count, 1)
    return avg_loss, global_step


@torch.no_grad()
def sample_images(model, reverse_diffusion, config, num_samples=64, class_labels=None):
    """Generate sample images."""
    model.eval()
    
    shape = (
        num_samples,
        config['model']['in_channels'],
        config['model']['image_size'],
        config['model']['image_size'],
    )
    
    guidance_scale = config.get('sampling', {}).get('guidance_scale', 1.0)
    
    # Generate samples
    samples = reverse_diffusion.p_sample_loop(
        shape,
        class_labels=class_labels,
        guidance_scale=guidance_scale,
        progress=True,
    )
    
    return samples


def main():
    """Train a diffusion model with the specified configuration."""
    parser = argparse.ArgumentParser(description='Train diffusion model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update log directory with experiment name
    config['logging']['log_dir'] = os.path.join('./experiments', args.experiment)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['logging']['log_dir'], 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config['logging']['log_dir'], 'samples'), exist_ok=True)
    
    # Save config
    with open(os.path.join(config['logging']['log_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Save reproducibility metadata
    import subprocess
    import hashlib
    metadata_path = os.path.join(config['logging']['log_dir'], 'run_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("Run Metadata\n\n")
        
        # Git commit hash
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                               stderr=subprocess.DEVNULL).decode('ascii').strip()
            f.write(f"Git commit: {git_hash}\n")
        except:
            f.write("Git commit: N/A (not a git repository)\n")
        
        # PyTorch version
        f.write(f"PyTorch version: {torch.__version__}\n")
        
        # CUDA version
        if torch.cuda.is_available():
            f.write(f"CUDA version: {torch.version.cuda}\n")
        else:
            f.write("CUDA version: N/A (CPU only)\n")
        
        # Config hash
        config_str = yaml.dump(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        f.write(f"Config SHA256: {config_hash}\n")
        
        f.write(f"\nExperiment: {args.experiment}\n")
        f.write(f"Config file: {args.config}\n")
    
    # Setup logger
    logger = setup_logger(config['logging']['log_dir'])
    logger.info(f"Starting experiment: {args.experiment}")
    logger.info(f"Config: {config}")
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataset
    train_dataset, _ = get_dataset(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
    )
    logger.info(f"Dataset size: {len(train_dataset)}")
    
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
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create diffusion process
    betas = get_beta_schedule(
        config['diffusion']['beta_schedule'],
        config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
    )
    diffusion_params = compute_diffusion_params(betas)
    forward_diffusion = ForwardDiffusion(diffusion_params, device)
    reverse_diffusion = ReverseDiffusion(model, diffusion_params, device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Create EMA
    ema = EMA(model, decay=config['training']['ema_decay'], device=device)
    
    # Create gradient scaler for mixed precision training
    use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed precision training (AMP) enabled")
    
    # TensorBoard writer
    writer = None
    if config['logging']['use_tensorboard']:
        writer = SummaryWriter(log_dir=config['logging']['log_dir'])
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(args.resume, model, optimizer, ema, device)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train
        avg_loss, global_step = train_epoch(
            model, train_loader, optimizer, forward_diffusion,
            device, epoch, writer, global_step, config, scaler
        )
        
        logger.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")
        
        # Update EMA
        ema.update(model)
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['logging']['log_dir'],
                'checkpoints',
                f'checkpoint_epoch_{epoch}.pt'
            )
            save_checkpoint(checkpoint_path, model, optimizer, ema, epoch, global_step)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(
                    config['logging']['log_dir'],
                    'checkpoints',
                    'best.pt'
                )
                save_checkpoint(best_path, model, optimizer, ema, epoch, global_step)
                logger.info(f"Saved best model: {best_path}")
        
        # Generate samples
        if (epoch + 1) % config['training']['sample_interval'] == 0:
            logger.info("Generating samples...")
            
            # Use EMA model for sampling
            ema.apply_shadow(model)
            
            # Generate unconditional samples
            num_samples = 64
            if config['conditioning']['enabled']:
                # Generate samples for each class
                num_classes = config['conditioning']['num_classes']
                samples_per_class = num_samples // num_classes
                # Adjust to be exactly divisible
                num_samples = samples_per_class * num_classes
                class_labels = torch.arange(num_classes, device=device)
                class_labels = class_labels.repeat_interleave(samples_per_class)
            else:
                class_labels = None
            
            samples = sample_images(model, reverse_diffusion, config, num_samples, class_labels)
            
            # Save samples
            from torchvision.utils import save_image
            save_path = os.path.join(
                config['logging']['log_dir'],
                'samples',
                f'samples_epoch_{epoch}.png'
            )
            save_image(samples, save_path, nrow=8, normalize=True, value_range=(-1, 1))
            logger.info(f"Saved samples: {save_path}")
            
            # Log to tensorboard
            if writer is not None:
                writer.add_images('samples', samples, epoch, dataformats='NCHW')
            
            # Restore original model
            ema.restore(model)
    
    logger.info("Training complete!")
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
