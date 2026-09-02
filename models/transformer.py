"""
Complete Decoder-Only Transformer Language Model in Pure PyTorch.
Integrates Token Embeddings, RMSNorm Pre-Normalization, GQA + RoPE,
SwiGLU FFN, Residual Connections, and KV Cache Autoregressive Generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from config import ModelConfig
from .rope import precompute_freqs_cis
from .layers import RMSNorm, SwiGLUFFN
from .attention import GroupedQueryAttention, KVCache


class TransformerBlock(nn.Module):
    """Single Decoder Layer in the Autoregressive Transformer."""

    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.head_dim

        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = GroupedQueryAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dropout=config.dropout
        )
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = SwiGLUFFN(
            dim=config.dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: Optional[int] = None,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-normalization residual block for Attention
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis=freqs_cis,
            start_pos=start_pos,
            kv_cache=kv_cache,
            mask=mask
        )
        # Pre-normalization residual block for Feed-Forward Network
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TransformerLM(nn.Module):
    """
    Complete Autoregressive Language Model constructed directly in pure PyTorch.
    No high-level wrappers.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.layers = nn.ModuleList([
            TransformerBlock(layer_id=i, config=config)
            for i in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Precompute RoPE frequency tensor
        freqs_cis = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_seq_len * 2,
            theta=config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: Optional[int] = 0,
        kv_caches: Optional[List[KVCache]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass over token sequence.
        
        Args:
            tokens: Int tensor of shape [batch_size, seq_len]
            start_pos: Position index for KV caching during inference
            kv_caches: Optional list of KVCache objects (one per layer)
            mask: Attention mask
            
        Returns:
            logits: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Retrieve RoPE slice for current token sequence window
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h = layer(
                h,
                freqs_cis=freqs_cis,
                start_pos=start_pos,
                kv_cache=cache,
                mask=mask
            )

        h = self.norm(h)
        logits = self.output(h)
        return logits.float()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 0.9,
        eos_id: int = 3
    ) -> torch.Tensor:
        """
        Fast autoregressive sequence generation using KV Cache and Nucleus (top-p) sampling.
        
        Args:
            prompt_ids: [1, prompt_len] input tensor of token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Softmax sampling temperature
            top_p: Nucleus sampling probability threshold
            eos_id: End of sequence token ID
            
        Returns:
            generated_ids: Tensor of token IDs including prompt and generated completion
        """
        self.eval()
        device = prompt_ids.device
        bsz, prompt_len = prompt_ids.shape

        # Initialize KV Cache per layer
        kv_caches = [
            KVCache(
                max_batch_size=bsz,
                max_seq_len=prompt_len + max_new_tokens,
                n_kv_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                dtype=self.tok_embeddings.weight.dtype,
                device=device
            )
            for _ in range(self.config.n_layers)
        ]

        # Prefill stage: Process full prompt to populate KV Cache
        logits = self.forward(prompt_ids, start_pos=0, kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :]

        generated = prompt_ids.clone()
        cur_pos = prompt_len

        for _ in range(max_new_tokens):
            if temperature > 0.0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                
                # Apply Top-P (Nucleus) Filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens with cumulative probability above top_p threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices right to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_id).all():
                break

            # Decode stage: Step forward single token using updated KV Cache
            logits = self.forward(next_token, start_pos=cur_pos, kv_caches=kv_caches)
            next_token_logits = logits[:, -1, :]
            cur_pos += 1

        return generated
