"""
Core neural layers for Decoder-Only Transformer architecture.
Includes Root Mean Square Layer Normalization (RMSNorm) and
SwiGLU Feed-Forward Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Provides faster pre-normalization without mean subtraction overhead.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (Gated Linear Unit with Swish/SiLU activation).
    SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None
    ):
        super().__init__()

        if hidden_dim is None:
            # Default SwiGLU hidden dim calculation: ~8/3 * dim rounded to multiple_of
            hidden_dim = int(2 * (4 * dim) / 3)
            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated computation: Swish(W_gate(x)) * W_up(x)
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
