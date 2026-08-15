"""
Dataset loading and preprocessing for language modeling.
Handles tokenization, batching, and sequence preparation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List


class TextDataset(Dataset):
    """
    Language modeling dataset that handles tokenization and batching.
    
    For causal language modeling, we need:
    - Input: tokens[:-1]
    - Target: tokens[1:]
    
    This shifting is handled in the collate function.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        stride: int = 256,
        min_length: int = 2,
        is_validation: bool = False,
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer with encode/decode methods
            max_length: Maximum sequence length
            stride: Stride for sliding window when processing long texts
            min_length: Minimum sequence length to ensure valid target generation
            is_validation: Whether this is a validation dataset
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.min_length = min_length
        self.is_validation = is_validation
        
        self.examples = []
        for text in texts:
            token_ids = tokenizer.encode(text)
            
            if self.is_validation and len(token_ids) < self.min_length:
                continue
            
            if len(token_ids) <= max_length:
                self.examples.append(token_ids)
            else:
                for i in range(0, len(token_ids) - max_length + 1, stride):
                    chunk = token_ids[i:i + max_length]
                    if len(chunk) == max_length:
                        self.examples.append(chunk)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single tokenized sequence.
        
        Returns:
            Dictionary with 'input_ids' tensor of shape (seq_len,)
        """
        token_ids = self.examples[idx]
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching sequences.
    
    Handles:
    - Padding sequences to same length
    - Creating attention masks
    - Shifting for causal LM (input vs target)
    
    Args:
        batch: List of examples from dataset
        pad_token_id: Token ID to use for padding
    
    Returns:
        Dictionary with:
            - input_ids: (batch_size, seq_len) - input tokens
            - labels: (batch_size, seq_len) - target tokens (shifted)
            - attention_mask: (batch_size, seq_len) - mask for padding
    """
    input_ids = [item['input_ids'] for item in batch]
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    attention_masks = []
    
    for ids in input_ids:
        padding_length = max_len - len(ids)
        padded_ids = torch.cat([
            ids,
            torch.full((padding_length,), pad_token_id, dtype=torch.long)
        ])
        
        mask = torch.cat([
            torch.ones(len(ids), dtype=torch.long),
            torch.zeros(padding_length, dtype=torch.long)
        ])
        
        padded_input_ids.append(padded_ids)
        attention_masks.append(mask)
    
    input_ids_batch = torch.stack(padded_input_ids)
    attention_mask_batch = torch.stack(attention_masks)
    
    labels_batch = input_ids_batch.clone()
    labels_batch[attention_mask_batch == 0] = -100
    
    return {
        'input_ids': input_ids_batch,
        'labels': labels_batch,
        'attention_mask': attention_mask_batch,
    }


def create_dataloaders(
    train_texts: List[str],
    val_texts: List[str],
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 0,
) -> tuple:
    """
    Create train and validation dataloaders.
    
    Args:
        train_texts: Training text samples
        val_texts: Validation text samples
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of dataloader workers
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = TextDataset(train_texts, tokenizer, max_length, is_validation=False)
    val_dataset = TextDataset(val_texts, tokenizer, max_length, is_validation=True)
    
    train_loader = None
    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
            pin_memory=True,
        )
    
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
            pin_memory=True,
        )
    
    return train_loader, val_loader


def load_text_dataset(dataset_name: str = "wikitext-2", num_train: int = 1000, num_val: int = 100) -> tuple:
    """
    Load a real language-modeling dataset.

    Supported names:
    - wikitext-2: Hugging Face wikitext-2-raw-v1 train/validation splits
    - tinystories: TinyStories train/validation slices
    - openwebtext-small: first rows from OpenWebText train split, with a train/val slice
    """
    from datasets import load_dataset

    name = dataset_name.lower()
    if name in {"wikitext", "wikitext-2", "wikitext2"}:
        train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        train_texts = [item["text"] for item in train if len(item["text"].strip()) > 50]
        val_texts = [item["text"] for item in val if len(item["text"].strip()) > 50]
    elif name in {"tinystories", "tiny-stories"}:
        train_texts = []
        if num_train > 0:
            train = load_dataset("roneneldan/TinyStories", split=f"train[:{num_train}]")
            train_texts = [item["text"] for item in train if len(item["text"].strip()) > 50]
        val = load_dataset("roneneldan/TinyStories", split=f"validation[:{num_val}]")
        val_texts = [item["text"] for item in val if len(item["text"].strip()) > 50]
        return train_texts, val_texts
    elif name in {"openwebtext-small", "openwebtext"}:
        total = num_train + num_val
        ds = load_dataset("Skylion007/openwebtext", split=f"train[:{total}]")
        texts = [item["text"] for item in ds if len(item["text"].strip()) > 50]
        train_texts = texts[:num_train]
        val_texts = texts[num_train : num_train + num_val]
    else:
        raise ValueError(f"Unsupported dataset_name={dataset_name}")

    return train_texts[:num_train], val_texts[:num_val]


def load_sample_data(
    tokenizer=None,
    num_train: int = 1000,
    num_val: int = 100,
    dataset_name: str = "wikitext-2",
    allow_synthetic_fallback: bool = False,
    return_metadata: bool = False,
) -> tuple:
    """
    Load sample dataset for demonstration.
    Uses a small subset of a public dataset.
    
    Args:
        tokenizer: Tokenizer instance
        num_train: Number of training examples
        num_val: Number of validation examples
    
    Returns:
        (train_texts, val_texts)
    """
    metadata = {"dataset_name": dataset_name, "synthetic": False}
    try:
        train_texts, val_texts = load_text_dataset(dataset_name=dataset_name, num_train=num_train, num_val=num_val)
        if return_metadata:
            return train_texts, val_texts, metadata
        return train_texts, val_texts
        
    except Exception as e:
        if not allow_synthetic_fallback:
            raise RuntimeError(
                f"Could not load requested dataset '{dataset_name}'. "
                "Refusing to silently replace it with synthetic text. "
                "Pass allow_synthetic_fallback=True only for smoke tests."
            ) from e
        print(f"Could not load dataset: {e}")
        print("Using synthetic data for demonstration...")
        
        train_texts = [
            f"This is training example number {i}. " * 20
            for i in range(num_train)
        ]
        val_texts = [
            f"This is validation example number {i}. " * 20
            for i in range(num_val)
        ]
        
        metadata = {"dataset_name": dataset_name, "synthetic": True, "fallback_error": str(e)}
        if return_metadata:
            return train_texts, val_texts, metadata
        return train_texts, val_texts
