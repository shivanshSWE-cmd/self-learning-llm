"""
Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) with KV Caching
and FlashAttention-2 support using PyTorch F.scaled_dot_product_attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .rope import apply_rotary_emb


class KVCache:
    """
    Key-Value Cache container for fast autoregressive generation.
    Stores past Key and Value activation states across sequence generation steps.
    """

    def __init__(self, max_batch_size: int, max_seq_len: int, n_kv_heads: int, head_dim: int, dtype=torch.bfloat16, device="cpu"):
        self.cache_k = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim), dtype=dtype, device=device)
        self.cache_v = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim), dtype=dtype, device=device)
        self.seq_len = 0

    def update(self, start_pos: int, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates cache at start_pos with new Key and Value tensors.
        xk, xv shapes: [bsz, seq_len, n_kv_heads, head_dim]
        """
        bsz, seqlen, _, _ = xk.shape
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        return keys, values

    def reset(self):
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.seq_len = 0


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands Key/Value tensor heads to match the number of Query heads for GQA.
    Input shape: [batch_size, seq_len, n_kv_heads, head_dim]
    Output shape: [batch_size, seq_len, n_kv_heads * n_rep, head_dim]
    """
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seqlen, n_kv_heads * n_rep, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) / Multi-Query Attention (MQA) Module.
    Integrates RoPE position embeddings, KV Caching, and FlashAttention execution.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = n_heads // n_kv_heads
        self.dropout = dropout

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: Optional[int] = None,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for GQA.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            freqs_cis: Precomputed RoPE complex frequencies for positions
            start_pos: Starting index position for inference generation (None during pre-training)
            kv_cache: Optional KVCache instance for autoregressive decoding
            mask: Causal attention mask [seq_len, seq_len] or boolean tensor
        """
        bsz, seqlen, _ = x.shape

        # Linear projections
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # Reshape to [bsz, seqlen, heads, head_dim]
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply Rotary Position Embedding
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Handle KV Cache during step-by-step inference
        if kv_cache is not None and start_pos is not None:
            keys, values = kv_cache.update(start_pos, xk, xv)
        else:
            keys, values = xk, xv

        # Repeat KV heads for Grouped-Query Attention if n_heads != n_kv_heads
        keys = repeat_kv(keys, self.num_queries_per_kv)     # [bsz, cache_len, n_heads, head_dim]
        values = repeat_kv(values, self.num_queries_per_kv) # [bsz, cache_len, n_heads, head_dim]

        # Transpose to PyTorch SDPA format: [bsz, n_heads, seqlen, head_dim]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Utilize PyTorch's scaled_dot_product_attention (dispatches to FlashAttention-2 when available)
        is_causal = (mask is None and start_pos is None and seqlen > 1)
        
        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask if not is_causal else None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        # Transpose back to [bsz, seqlen, n_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
