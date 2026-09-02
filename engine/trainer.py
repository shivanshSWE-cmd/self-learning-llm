"""
Distributed Training Engine supporting PyTorch DDP/FSDP, Automatic Mixed Precision (AMP),
Gradient Accumulation, Cosine Annealing Learning Rate Schedule with Warmup, and Model Checkpointing.
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict

from config import TrainConfig
from .losses import compute_causal_ce_loss


def get_cosine_schedule_with_warmup_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr: float
) -> float:
    """Calculates Learning Rate at step using Linear Warmup + Cosine Annealing."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


class DistributedTrainer:
    """
    Production-grade distributed trainer supporting DDP / single-device fallback,
    AMP bfloat16 mixed precision, gradient scaling & accumulation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        device: torch.device,
        is_distributed: bool = False
    ):
        self.model = model
        self.config = train_config
        self.device = device
        self.is_distributed = is_distributed
        self.rank = dist.get_rank() if is_distributed else 0

        # Move model to assigned hardware device
        self.model.to(self.device)

        # Setup DDP wrapper if in distributed multi-GPU mode
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.rank])

        # Configure AdamW Optimizer with weight decay decay separation
        self.optimizer = self._configure_optimizer()

        # Automatic Mixed Precision (AMP) Scaler
        amp_dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        self.amp_dtype = amp_dtype_map.get(self.config.amp_dtype, torch.bfloat16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.use_amp and self.amp_dtype == torch.float16))

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Separates weight decay for 2D weight matrices from 1D bias/norm vectors."""
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )
        return optimizer

    def train_epoch(self, dataloader: DataLoader, step_start: int = 0) -> int:
        """Executes pre-training / fine-tuning step loop over dataloader."""
        self.model.train()
        step = step_start
        total_loss = 0.0
        start_time = time.time()

        self.optimizer.zero_grad(set_to_none=True)

        for micro_step, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            # Update Learning Rate according to Cosine schedule
            lr = get_cosine_schedule_with_warmup_lr(
                step=step,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                base_lr=self.config.learning_rate,
                min_lr=self.config.min_lr
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass under Automatic Mixed Precision (AMP)
            with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=self.config.use_amp and self.device.type == "cuda"):
                logits = self.model(x)
                loss = compute_causal_ce_loss(logits, y)
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.config.gradient_accumulation_steps

            # Optimizer Step on Gradient Accumulation boundary
            if (micro_step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                step += 1

                if self.rank == 0 and step % self.config.eval_interval == 0:
                    elapsed = time.time() - start_time
                    avg_loss = total_loss / (micro_step + 1)
                    tok_per_sec = (x.numel() * self.config.gradient_accumulation_steps) / max(0.001, elapsed)
                    print(f"Step {step}/{self.config.max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | Speed: {tok_per_sec:.1f} tok/s")
                    start_time = time.time()

                if step >= self.config.max_steps:
                    break

        return step

    def save_checkpoint(self, filepath: str, meta: Optional[Dict] = None):
        """Saves model state, optimizer state, and metadata checkpoint."""
        if self.rank != 0:
            return
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {
            "model_state": raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": raw_model.config,
            "meta": meta or {}
        }
        torch.save(checkpoint, filepath)
        print(f"Saved model checkpoint to: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Loads checkpoint into model and optimizer."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Loaded checkpoint from: {filepath}")
