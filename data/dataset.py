"""
Memory-mapped PyTorch Dataset and DataLoader routines.
Supports zero-copy dataset loading using np.memmap for massive text corpora,
dynamic window slicing, and causal attention mask creation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class MemmapDataset(Dataset):
    """
    High-performance, zero-copy PyTorch Dataset leveraging NumPy memory-mapping.
    Chunks continuous 1D token ID arrays into contiguous sequences of fixed length.
    """

    def __init__(self, bin_file_path: str, seq_len: int, dtype=np.uint16):
        super().__init__()
        self.bin_file_path = bin_file_path
        self.seq_len = seq_len
        self.dtype = dtype

        if not os.path.exists(bin_file_path):
            raise FileNotFoundError(f"Binary dataset file not found at: {bin_file_path}")

        # Compute dataset file size and token count
        file_size_bytes = os.path.getsize(bin_file_path)
        item_size = np.dtype(dtype).itemsize
        self.total_tokens = file_size_bytes // item_size

        # Open memory-map pointer
        self.data = np.memmap(bin_file_path, dtype=dtype, mode="r")
        # Compute total usable samples (each sample requires seq_len + 1 tokens for input + target)
        self.num_samples = (self.total_tokens - 1) // seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1

        # Fetch sequence slice [seq_len + 1]
        chunk = np.array(self.data[start_idx:end_idx], dtype=np.int64)

        # Input x is token 0..seq_len-1, target y is token 1..seq_len
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Creates a causal (lower-triangular) boolean mask for self-attention.
    True values indicate positions that are masked out (cannot be attended to).
    Shape: [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask


def create_dataloader(
    bin_file_path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = False
) -> DataLoader:
    """Factory helper to build a PyTorch DataLoader wrapping MemmapDataset."""
    dataset = MemmapDataset(bin_file_path=bin_file_path, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


def text_to_memmap(text: str, tokenizer, bin_file_path: str, dtype=np.uint16):
    """Helper script to encode raw text corpus and save to uint16 memory-mapped binary file."""
    os.makedirs(os.path.dirname(os.path.abspath(bin_file_path)), exist_ok=True)
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    arr = np.array(token_ids, dtype=dtype)
    
    with open(bin_file_path, "wb") as f:
        f.write(arr.tobytes())
    return len(arr)
