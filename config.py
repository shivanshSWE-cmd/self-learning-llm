"""
Configuration parameters for the Self-Learning Autoregressive Large Language Model.
Includes specifications for model architecture, data pipeline, training, DPO alignment,
EWC catastrophic forgetting safeguards, and self-improvement loops.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Transformer Architecture Hyperparameters."""
    vocab_size: int = 32000
    dim: int = 512              # Hidden dimension size (d_model)
    n_layers: int = 8           # Number of transformer decoder blocks
    n_heads: int = 8            # Number of Query attention heads
    n_kv_heads: Optional[int] = 2  # Number of Key/Value heads for GQA (n_heads // n_kv_heads is group size)
    multiple_of: int = 256      # SwiGLU hidden dim dimension alignment factor
    ffn_dim_multiplier: Optional[float] = None  # Custom multiplier for FFN hidden dim
    norm_eps: float = 1e-5      # RMSNorm epsilon
    rope_theta: float = 10000.0 # Base frequency parameter for RoPE
    max_seq_len: int = 1024     # Maximum context window length
    dropout: float = 0.0        # Dropout rate (0.0 recommended for LLM pre-training)
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        self.head_dim = self.dim // self.n_heads
        assert self.head_dim * self.n_heads == self.dim, "dim must be divisible by n_heads"


@dataclass
class DataConfig:
    """Dataset & Data Loading Settings."""
    raw_data_path: str = "data/corpus.txt"
    bin_data_path: str = "data/dataset.bin"
    tokenizer_path: str = "tokenizer/vocab.json"
    vocab_size: int = 32000
    seq_len: int = 1024
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class TrainConfig:
    """Pre-training & Optimization Parameters."""
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 1000
    gradient_accumulation_steps: int = 2
    use_amp: bool = True
    amp_dtype: str = "bfloat16" # "bfloat16" or "float16"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 250
    eval_interval: int = 50


@dataclass
class DPOConfig:
    """Direct Preference Optimization Settings."""
    beta: float = 0.1           # KL divergence regularization parameter in DPO loss
    learning_rate: float = 5e-6 # Typically lower than pretraining LR
    batch_size: int = 4
    max_prompt_len: int = 512
    max_response_len: int = 512


@dataclass
class EWCConfig:
    """Elastic Weight Consolidation (Continual Learning Safeguard) Settings."""
    ewc_lambda: float = 500.0   # Regularization penalty weighting
    num_fisher_samples: int = 256 # Sample count from replay buffer to estimate Fisher matrix
    fisher_epsilon: float = 1e-8


@dataclass
class SelfLearningConfig:
    """Autonomous Self-Improvement Loop Settings."""
    num_cycles: int = 3               # Number of self-improvement iterations
    prompts_per_cycle: int = 20       # Seed prompts to generate synthetic responses for
    candidates_per_prompt: int = 4   # Candidate completions to evaluate per prompt
    temperature: float = 0.8
    top_p: float = 0.9
    dpo_epochs_per_cycle: int = 2
    replay_ratio: float = 0.2         # Ratio of baseline pre-training data retained in replay buffer
