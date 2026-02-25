"""
GPT Model - Decoder-only Transformer Language Model.

This is the main model class that combines all components:
- Token + Positional Embeddings
- Stack of Transformer Blocks
- Language Modeling Head

Architecture follows GPT-2 / GPT-3 style.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .embeddings import TransformerEmbedding
from .transformer_block import TransformerBlock
from .utils import initialize_weights, count_parameters


class GPTConfig:
    """
    Configuration for GPT model.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        bias: bool = True,
        use_sinusoidal_pos: bool = False,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            max_seq_len: Maximum sequence length
            d_model: Model dimension (embedding size)
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            use_sinusoidal_pos: Use sinusoidal vs learned positional embeddings
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.bias = bias
        self.use_sinusoidal_pos = use_sinusoidal_pos
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'bias': self.bias,
            'use_sinusoidal_pos': self.use_sinusoidal_pos,
        }


class GPT(nn.Module):
    """
    GPT: Generative Pre-trained Transformer (Decoder-only).
    
    Architecture:
    1. Token + Positional Embeddings
    2. Stack of N Transformer Blocks
    3. Final LayerNorm
    4. Language Modeling Head (projects to vocabulary)
    
    For training: Returns logits and loss
    For inference: Supports autoregressive generation with KV caching
    """
    
    def __init__(self, config: GPTConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # 1. Embeddings (token + positional)
        self.embedding = TransformerEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            use_sinusoidal=config.use_sinusoidal_pos,
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                bias=config.bias,
            )
            for _ in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.token_embedding.embedding.weight
        
        self.apply(lambda module: initialize_weights(module, init_std=0.02))
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> dict:
        """
        Forward pass through GPT model.
        
        Args:
            input_ids: Token IDs of shape (B, T)
            attention_mask: Optional mask of shape (B, T)
            labels: Optional labels for computing loss, shape (B, T)
            kv_caches: Optional list of (K, V) caches for each layer
            use_cache: Whether to return KV caches
            start_pos: Starting position for positional embeddings (for generation)
        
        Returns:
            Dictionary containing:
                - logits: Output logits of shape (B, T, vocab_size)
                - loss: Optional loss if labels provided
                - kv_caches: Optional list of (K, V) caches if use_cache=True
        """
        B, T = input_ids.shape
        
        x = self.embedding(input_ids, start_pos=start_pos)
        
        new_kv_caches = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None
            
            x, new_kv_cache = block(
                x,
                attention_mask=attention_mask,
                kv_cache=layer_kv_cache,
                use_cache=use_cache,
            )
            
            if use_cache:
                new_kv_caches.append(new_kv_cache)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'kv_caches': new_kv_caches,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Starting tokens of shape (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Keep tokens with cumulative probability >= top_p
            use_cache: Use KV caching for efficiency
        
        Returns:
            Generated token IDs of shape (B, T + max_new_tokens)
        """
        self.eval()
        
        kv_caches = None
        
        for _ in range(max_new_tokens):
            if use_cache and kv_caches is not None:
                input_ids_step = input_ids[:, -1:]
                start_pos = input_ids.shape[1] - 1
            else:
                input_ids_step = input_ids
                start_pos = 0
            
            outputs = self.forward(
                input_ids_step,
                kv_caches=kv_caches,
                use_cache=use_cache,
                start_pos=start_pos,
            )
            
            logits = outputs['logits']
            kv_caches = outputs['kv_caches']
            
            logits = logits[:, -1, :]
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters."""
        return count_parameters(self, trainable_only)
