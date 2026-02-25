"""
Embedding layers for transformer model.

Includes:
- Token embeddings
- Positional embeddings
- Combined embedding with dropout
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.
    Maps token IDs to dense vectors.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Token IDs of shape (B, T)
        
        Returns:
            Embeddings of shape (B, T, d_model)
        """
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """
    Learned positional embeddings.
    
    Alternative to sinusoidal embeddings used in original Transformer.
    GPT-2 and GPT-3 use learned positional embeddings.
    """
    
    def __init__(self, max_seq_len: int, d_model: int):
        """
        Args:
            max_seq_len: Maximum sequence length
            d_model: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, d_model) or (B, T)
            start_pos: Starting position (for KV cache during generation)
        
        Returns:
            Positional embeddings of shape (B, T, d_model)
        """
        B, T = x.shape[:2]
        
        # Create position indices
        positions = torch.arange(start_pos, start_pos + T, device=x.device)
        
        # Get positional embeddings
        pos_emb = self.embedding(positions)  # (T, d_model)
        
        # Broadcast to batch dimension
        return pos_emb.unsqueeze(0)  # (1, T, d_model)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings from "Attention is All You Need".
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    These are fixed (not learned) and can extrapolate to longer sequences.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """
        Args:
            d_model: Embedding dimension
            max_seq_len: Maximum sequence length to precompute
        """
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, d_model) or (B, T)
            start_pos: Starting position (for KV cache during generation)
        
        Returns:
            Positional embeddings of shape (1, T, d_model)
        """
        T = x.shape[1]
        return self.pe[:, start_pos:start_pos + T, :]


class TransformerEmbedding(nn.Module):
    """
    Combined token + positional embeddings for transformer.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
        use_sinusoidal: bool = False,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_sinusoidal: Use sinusoidal (True) or learned (False) positional embeddings
        """
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        
        if use_sinusoidal:
            self.pos_embedding = SinusoidalPositionalEmbedding(d_model, max_seq_len)
        else:
            self.pos_embedding = PositionalEmbedding(max_seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, token_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            token_ids: Token IDs of shape (B, T)
            start_pos: Starting position (for KV cache during generation)
        
        Returns:
            Combined embeddings of shape (B, T, d_model)
        """
        token_emb = self.token_embedding(token_ids)
        pos_emb = self.pos_embedding(token_ids, start_pos)
        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)
        
        return embeddings
