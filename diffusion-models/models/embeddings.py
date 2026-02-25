"""
Embedding modules for timesteps and class labels.

Implements sinusoidal positional embeddings for continuous timesteps
and learned embeddings for discrete class labels.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal timestep embeddings as used in "Attention is All You Need".
    
    Maps continuous timestep values to high-dimensional embeddings that
    preserve temporal relationships through periodic functions.
    
    Args:
        dim: Embedding dimension (must be even)
    """
    
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.dim = dim
        
    def forward(self, time):
        """
        Args:
            time: (batch_size,) tensor of timestep values
            
        Returns:
            embeddings: (batch_size, dim) tensor of sinusoidal embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        
        # Compute frequency scaling: 10000^(-2i/d) for i in [0, d/2)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Compute sinusoidal embeddings: [sin(t*w_i), cos(t*w_i)]
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class TimestepEmbedding(nn.Module):
    """
    Projects sinusoidal timestep embeddings to model dimension.
    
    Applies a two-layer MLP with SiLU activation to transform
    raw sinusoidal embeddings into a representation suitable for
    injection into the UNet.
    
    Args:
        time_embed_dim: Input dimension (from sinusoidal embeddings)
        model_dim: Output dimension (model's hidden dimension)
    """
    
    def __init__(self, time_embed_dim, model_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        
    def forward(self, time_emb):
        """
        Args:
            time_emb: (batch_size, time_embed_dim) sinusoidal embeddings
            
        Returns:
            (batch_size, model_dim) projected embeddings
        """
        return self.mlp(time_emb)


class ClassEmbedding(nn.Module):
    """
    Learned embeddings for class labels with support for unconditional generation.
    
    Implements classifier-free guidance by learning a special "null" class
    that represents unconditional generation.
    
    Args:
        num_classes: Number of classes in dataset
        embed_dim: Embedding dimension
        dropout_prob: Probability of using null class during training
    """
    
    def __init__(self, num_classes, embed_dim, dropout_prob=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
        # +1 for the unconditional "null" class
        self.embedding = nn.Embedding(num_classes + 1, embed_dim)
        self.null_class_idx = num_classes
        
    def forward(self, class_labels, force_drop=False):
        """
        Args:
            class_labels: (batch_size,) tensor of class indices [0, num_classes)
            force_drop: If True, always use null class (for unconditional sampling)
            
        Returns:
            (batch_size, embed_dim) class embeddings
        """
        if self.training and self.dropout_prob > 0:
            # Randomly replace some labels with null class for classifier-free guidance
            mask = torch.rand(class_labels.shape[0], device=class_labels.device) < self.dropout_prob
            class_labels = torch.where(mask, self.null_class_idx, class_labels)
        
        if force_drop:
            # Use null class for all samples (unconditional generation)
            class_labels = torch.full_like(class_labels, self.null_class_idx)
            
        return self.embedding(class_labels)
