"""
Rotary Position Embeddings (RoPE) implementation in pure PyTorch.
Computes frequency matrices and rotates Query/Key head representations.
"""

import torch
import torch.nn as nn
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes complex exponential frequency tensor for RoPE.
    
    Args:
        dim: Dimension of individual attention head (head_dim).
        end: Maximum sequence length.
        theta: Base frequency hyperparameter.
        
    Returns:
        freqs_cis: Complex tensor of shape [end, dim // 2]
    """
    # Inv frequencies: 1.0 / (theta ** (2i / dim)) for i in 0..dim//2 - 1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    # Outer product -> [end, dim // 2]
    freqs = torch.outer(t, freqs)
    # Convert polar coordinates (r=1, theta=freqs) to complex cis = cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to Query and Key tensors.
    
    Args:
        xq: Query tensor of shape [batch_size, seq_len, n_heads, head_dim]
        xk: Key tensor of shape [batch_size, seq_len, n_kv_heads, head_dim]
        freqs_cis: Complex frequency tensor of shape [seq_len, head_dim // 2]
        
    Returns:
        xq_out, xk_out with rotated position embeddings applied.
    """
    # Reshape xq and xk to pair adjacent head dimensions into complex numbers
    # xq: [B, T, H, D] -> complex xq_: [B, T, H, D // 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs_cis from [T, D // 2] to [1, T, 1, D // 2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Complex multiplication rotates phases
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
