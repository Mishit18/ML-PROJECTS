"""
Tokenizer wrapper for text encoding/decoding.
Uses HuggingFace tokenizers but with explicit control over padding and special tokens.
"""

from transformers import GPT2TokenizerFast
from typing import List, Union
import json
import os


class Tokenizer:
    """
    Wrapper around HuggingFace tokenizer with explicit interface.
    
    Key responsibilities:
    - Encode text to token IDs
    - Decode token IDs to text
    - Handle special tokens (BOS, EOS, PAD)
    - Provide vocabulary information
    """
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        """
        Initialize tokenizer.
        
        Args:
            tokenizer_name: Name of pretrained tokenizer or path to custom tokenizer
        """
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.eos_token_id
        self.vocab_size = len(self.tokenizer)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Single sequence or batch of token ID sequences
            skip_special_tokens: Whether to remove special tokens from output
        
        Returns:
            Decoded text string or list of strings
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode multiple texts.
        
        Args:
            texts: List of text strings
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            List of token ID sequences
        """
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def batch_decode(self, token_ids_batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """
        Decode multiple sequences.
        
        Args:
            token_ids_batch: List of token ID sequences
            skip_special_tokens: Whether to remove special tokens
        
        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(token_ids_batch, skip_special_tokens=skip_special_tokens)
    
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.tokenizer.save_pretrained(save_dir)
        
        metadata = {
            'vocab_size': self.vocab_size,
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id,
            'bos_token_id': self.bos_token_id,
        }
        with open(os.path.join(save_dir, 'tokenizer_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, load_dir: str):
        """Load tokenizer from directory."""
        tokenizer = cls.__new__(cls)
        tokenizer.tokenizer = GPT2TokenizerFast.from_pretrained(load_dir)
        
        with open(os.path.join(load_dir, 'tokenizer_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        tokenizer.vocab_size = metadata['vocab_size']
        tokenizer.pad_token_id = metadata['pad_token_id']
        tokenizer.eos_token_id = metadata['eos_token_id']
        tokenizer.bos_token_id = metadata['bos_token_id']
        
        return tokenizer
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


def create_tokenizer(tokenizer_name: str = "gpt2") -> Tokenizer:
    """
    Factory function to create tokenizer.
    
    Args:
        tokenizer_name: Name of tokenizer to load
    
    Returns:
        Tokenizer instance
    """
    return Tokenizer(tokenizer_name)
