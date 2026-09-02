"""
Experience Replay Buffer & Fisher Matrix Management for Continual Learning.
Prevents catastrophic forgetting during self-improvement DPO cycles by maintaining
baseline anchor datasets and updating EWC Fisher Information matrices.
"""

import copy
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from engine.losses import estimate_fisher_information


class BaselineReplayBuffer:
    """
    Maintains baseline anchor pre-training data and updates Fisher Information Matrix.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.anchor_parameters: Optional[Dict[str, torch.Tensor]] = None
        self.fisher_matrix: Optional[Dict[str, torch.Tensor]] = None

    def add_sample(self, input_ids: torch.Tensor, target_ids: torch.Tensor):
        """Adds a baseline dataset token sequence pair (x, y) to replay buffer."""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0) # Evict oldest sample FIFO
        self.buffer.append((input_ids.cpu(), target_ids.cpu()))

    def populate_from_dataloader(self, dataloader: DataLoader, max_samples: int = 500):
        """Fills replay buffer from baseline pre-training dataloader."""
        count = 0
        for x, y in dataloader:
            for i in range(x.size(0)):
                self.add_sample(x[i], y[i])
                count += 1
                if count >= max_samples:
                    return

    def get_dataloader(self, batch_size: int = 4) -> DataLoader:
        """Returns DataLoader over stored replay buffer samples."""
        class ReplayDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        dataset = ReplayDataset(self.buffer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def snapshot_anchor_model(self, model: nn.Module):
        """Freezes and stores a reference copy of optimal baseline model parameters theta_A*."""
        self.anchor_parameters = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def compute_fisher_matrix(
        self,
        model: nn.Module,
        num_samples: int = 256,
        device: torch.device = None
    ) -> Dict[str, torch.Tensor]:
        """Computes diagonal Fisher Information Matrix over current replay dataset."""
        replay_loader = self.get_dataloader(batch_size=4)
        self.fisher_matrix = estimate_fisher_information(
            model=model,
            dataloader=replay_loader,
            num_samples=num_samples,
            device=device
        )
        return self.fisher_matrix
