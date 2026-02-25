"""
Dataset utilities for loading and preprocessing image data.
"""

import torch
from torchvision import datasets, transforms
import os


def get_cifar10_transforms(image_size=32):
    """
    Get transforms for CIFAR-10 dataset.
    
    Normalizes images to [-1, 1] range as expected by diffusion models.
    
    Args:
        image_size: Target image size (CIFAR-10 is 32x32)
        
    Returns:
        transform: Composed transforms
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale to [-1, 1]
    ])
    return transform


def get_dataset(config):
    """
    Get dataset based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
    """
    dataset_name = config['dataset']['name'].lower()
    data_dir = config['dataset']['data_dir']
    image_size = config['model']['image_size']
    
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name == 'cifar10':
        transform = get_cifar10_transforms(image_size)
        
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        
        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset


class UnlabeledDataset(torch.utils.data.Dataset):
    """
    Wrapper to remove labels from a dataset (for unconditional training).
    
    Args:
        dataset: Original dataset with (image, label) pairs
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, 0  # Return dummy label
