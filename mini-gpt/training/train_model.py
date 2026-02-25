"""Training script for GPT model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import os
import sys
import argparse
import yaml
import matplotlib.pyplot as plt
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT, GPTConfig
from model.utils import set_seed, get_device
from tokenizer.tokenizer import create_tokenizer
from data.dataset import load_sample_data, create_dataloaders
from training.optimizer import configure_optimizers
from training.scheduler import get_cosine_schedule_with_warmup


logging.basicConfig(level=logging.INFO, format='%(message)s')


class Trainer:
    """Trainer for GPT model with mixed precision and gradient accumulation."""
    
    def __init__(self, model, train_loader, optimizer, scheduler, device, config):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        self.step = 0
        self.epoch = 0
        self.train_losses = []
        
        self.use_amp = config.get('use_amp', False) and device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.grad_accum_steps = config.get('grad_accum_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.log_interval = config.get('log_interval', 100)
        self.save_interval = config.get('save_interval', 1000)
    
    def train_step(self, batch):
        """Single training step."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        if self.use_amp:
            with autocast('cuda'):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss'] / self.grad_accum_steps
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss'] / self.grad_accum_steps
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.grad_accum_steps
    
    def optimizer_step(self):
        """Optimizer step with gradient clipping."""
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.scheduler is not None:
            self.scheduler.step()
    
    def train(self, num_epochs):
        """Main training loop."""
        logging.info(f"\nTraining for {num_epochs} epochs")
        logging.info(f"Device: {self.device} | Mixed Precision: {self.use_amp}")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            logging.info(f"\n{'='*60}\nEpoch {epoch + 1}/{num_epochs}\n{'='*60}")
            
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.optimizer_step()
                    self.step += 1
                    self.train_losses.append(loss)
                    
                    if self.step % self.log_interval == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        logging.info(f"Step {self.step:5d} | Loss: {loss:.4f} | LR: {lr:.2e}")
                    
                    if self.step % self.save_interval == 0:
                        self.save_checkpoint(f'checkpoint_step_{self.step}.pt')
            
            logging.info(f"End of Epoch {epoch + 1}")
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        self.save_checkpoint('model_final.pt')
        self.plot_training_curves()
        logging.info("\nTraining complete!")
    
    def save_checkpoint(self, filename):
        """Save training checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'config': self.config,
        }, path)
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        os.makedirs('experiments', exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig('experiments/train_loss.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train GPT model')
    parser.add_argument('--config', type=str, default='configs/small.yaml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(args.seed)
    device = get_device()
    
    tokenizer = create_tokenizer()
    train_texts, _ = load_sample_data(
        tokenizer,
        num_train=config['data']['num_train'],
        num_val=0,
    )
    
    train_loader, _ = create_dataloaders(
        train_texts, [], tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_seq_len'],
    )
    
    model_config = GPTConfig(**config['model'])
    model = GPT(model_config).to(device)
    
    optimizer = configure_optimizers(
        model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        device_type=device.type,
    )
    
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    num_warmup_steps = int(num_training_steps * config['training']['warmup_ratio'])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=config['training']['min_lr_ratio'],
    )
    
    trainer = Trainer(model, train_loader, optimizer, scheduler, device, config['training'])
    trainer.train(num_epochs=config['training']['num_epochs'])


if __name__ == '__main__':
    main()
