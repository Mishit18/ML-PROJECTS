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
import math
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT, GPTConfig
from model.utils import set_seed, get_device
from tokenizer.tokenizer import create_tokenizer
from data.dataset import load_sample_data, create_dataloaders
from training.optimizer import configure_optimizers
from training.scheduler import get_cosine_schedule_with_warmup
from training.evaluate import evaluate_model
from training.tracker import ExperimentTracker, cuda_memory_mb


logging.basicConfig(level=logging.INFO, format='%(message)s')


class Trainer:
    """Trainer for GPT model with mixed precision and gradient accumulation."""
    
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config, full_config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.full_config = full_config
        
        self.step = 0
        self.epoch = 0
        self.train_losses = []
        self.val_metrics = []
        self.tokens_trained = 0
        self.tracker = ExperimentTracker(
            output_dir=config.get('experiment_dir', 'experiments'),
            run_name=config.get('run_name', 'mini_gpt')
        )
        
        self.use_amp = config.get('use_amp', False) and device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.grad_accum_steps = config.get('grad_accum_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.log_interval = config.get('log_interval', 100)
        self.save_interval = config.get('save_interval', 1000)
        self.best_val_loss = float('inf')
        self.best_checkpoint_name = 'model_best.pt'
        self.target_perplexity = config.get('target_perplexity')
        self.should_stop = False
    
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
        self.tokens_trained += int(attention_mask[:, 1:].sum().item())
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
            
            epoch_start = time.perf_counter()
            for batch_idx, batch in enumerate(self.train_loader):
                step_start_tokens = self.tokens_trained
                step_start = time.perf_counter()
                loss = self.train_step(batch)
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.optimizer_step()
                    self.step += 1
                    self.train_losses.append(loss)
                    
                    if self.step % self.log_interval == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        elapsed = time.perf_counter() - step_start
                        step_tokens = self.tokens_trained - step_start_tokens
                        tokens_per_sec = step_tokens / elapsed if elapsed > 0 else 0.0
                        logging.info(f"Step {self.step:5d} | Train loss: {loss:.4f} | LR: {lr:.2e} | {tokens_per_sec:.1f} tok/s")
                        self.tracker.log({
                            'split': 'train',
                            'step': self.step,
                            'epoch': epoch + 1,
                            'train_loss': loss,
                            'validation_loss': '',
                            'perplexity': '',
                            'tokens_trained': self.tokens_trained,
                            'tokens_per_sec': tokens_per_sec,
                            'gpu_memory_mb': cuda_memory_mb(),
                            'config': self.full_config,
                        })

                    if self.val_loader is not None and self.step % self.config.get('eval_interval', 500) == 0:
                        metrics = evaluate_model(self.model, self.val_loader, self.device)
                        self.val_metrics.append({'step': self.step, **metrics})
                        self.maybe_save_best(metrics, epoch=epoch + 1)
                        logging.info(
                            f"Eval step {self.step:5d} | Val loss: {metrics['val_loss']:.4f} | "
                            f"PPL: {metrics['perplexity']:.2f} | {metrics['tokens_per_sec']:.1f} tok/s"
                        )
                        self.tracker.log({
                            'split': 'validation',
                            'step': self.step,
                            'epoch': epoch + 1,
                            'train_loss': loss,
                            'validation_loss': metrics['val_loss'],
                            'perplexity': metrics['perplexity'],
                            'tokens_trained': self.tokens_trained,
                            'tokens_per_sec': metrics['tokens_per_sec'],
                            'gpu_memory_mb': cuda_memory_mb(),
                            'config': self.full_config,
                        })
                        if self.target_perplexity is not None and metrics['perplexity'] <= self.target_perplexity:
                            logging.info(f"Target perplexity {self.target_perplexity:.2f} reached at step {self.step}.")
                            self.should_stop = True
                        self.model.train()
                        if self.should_stop:
                            break
                    
                    if self.step % self.save_interval == 0:
                        self.save_checkpoint(f'checkpoint_step_{self.step}.pt')
            if self.should_stop:
                logging.info("Stopping early after reaching target perplexity.")
                break
            
            logging.info(f"End of Epoch {epoch + 1}")
            if self.val_loader is not None:
                metrics = evaluate_model(self.model, self.val_loader, self.device)
                self.val_metrics.append({'step': self.step, **metrics})
                self.maybe_save_best(metrics, epoch=epoch + 1)
                epoch_tokens_sec = self.tokens_trained / max(1e-9, time.perf_counter() - epoch_start)
                logging.info(
                    f"Epoch {epoch + 1} validation | Val loss: {metrics['val_loss']:.4f} | "
                    f"PPL: {metrics['perplexity']:.2f} | Train throughput: {epoch_tokens_sec:.1f} tok/s"
                )
                self.tracker.log({
                    'split': 'epoch',
                    'step': self.step,
                    'epoch': epoch + 1,
                    'train_loss': self.train_losses[-1] if self.train_losses else '',
                    'validation_loss': metrics['val_loss'],
                    'perplexity': metrics['perplexity'],
                    'tokens_trained': self.tokens_trained,
                    'tokens_per_sec': epoch_tokens_sec,
                    'gpu_memory_mb': cuda_memory_mb(),
                    'config': self.full_config,
                })
                self.model.train()
                if self.target_perplexity is not None and metrics['perplexity'] <= self.target_perplexity:
                    logging.info(f"Target perplexity {self.target_perplexity:.2f} reached at epoch {epoch + 1}.")
                    self.should_stop = True
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            if self.should_stop:
                logging.info("Stopping early after reaching target perplexity.")
                break
        
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
            'val_metrics': self.val_metrics,
            'tokens_trained': self.tokens_trained,
            'best_val_loss': self.best_val_loss,
            'best_checkpoint_name': self.best_checkpoint_name if self.best_val_loss < float('inf') else None,
            'config': self.config,
            'full_config': self.full_config,
        }, path)
    
    def maybe_save_best(self, metrics, epoch):
        """Save best checkpoint by validation loss."""
        if metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            logging.info(
                f"New best validation loss: {metrics['val_loss']:.4f} "
                f"(PPL {metrics['perplexity']:.2f}) at epoch {epoch}, step {self.step}"
            )
            self.save_checkpoint(self.best_checkpoint_name)
    
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
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['wikitext-2', 'tinystories', 'openwebtext-small'])
    parser.add_argument('--allow-synthetic-fallback', action='store_true',
                        help='Allow synthetic fallback only for smoke testing when dataset download fails')
    parser.add_argument('--init-from-checkpoint', type=str, default=None,
                        help='Initialize model weights from a checkpoint but start a fresh optimizer/schedule')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(args.seed)
    device = get_device()
    
    tokenizer = create_tokenizer()
    train_texts, val_texts, data_metadata = load_sample_data(
        tokenizer,
        num_train=config['data']['num_train'],
        num_val=config['data'].get('num_val', 100),
        dataset_name=args.dataset or config['data'].get('dataset_name', 'wikitext-2'),
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        return_metadata=True,
    )
    config['data']['synthetic'] = data_metadata['synthetic']
    
    train_loader, val_loader = create_dataloaders(
        train_texts, val_texts, tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_seq_len'],
    )
    
    model_config = GPTConfig(**config['model'])
    model = GPT(model_config).to(device)
    if args.init_from_checkpoint:
        checkpoint = torch.load(args.init_from_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Initialized model weights from {args.init_from_checkpoint}")
    
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
    
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, device, config['training'], config)
    trainer.train(num_epochs=config['training']['num_epochs'])


if __name__ == '__main__':
    main()
